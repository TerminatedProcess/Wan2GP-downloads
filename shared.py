#!/usr/bin/env python3
"""
Shared utilities for WanGP Model Downloader.
Used by both downloader_st.py (Streamlit UI) and hfqueue.py (queue processor).
"""

import json
import yaml
import time
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
from urllib.parse import urlparse

import requests
import httpx

# Queue database path (shared between Streamlit UI and queue processor)
QUEUE_DB_PATH = "hfcache.db"

DEFAULT_CONFIG = {
    "wan2gp_directory": "../Wan2GP-mryan",
    "bandwidth_limit_kb": 90000,
    "hub_directory": "",
}


class BandwidthLimitedSession(requests.Session):
    """Requests session with bandwidth limiting"""

    def __init__(self, max_bytes_per_second: Optional[int] = None):
        super().__init__()
        self.max_bytes_per_second = max_bytes_per_second

    def request(self, method, url, **kwargs):
        response = super().request(method, url, **kwargs)

        if (self.max_bytes_per_second and
            hasattr(response, 'headers') and
            response.headers.get('content-length') and
            int(response.headers.get('content-length', 0)) > 1024 * 1024):

            original_iter_content = response.iter_content

            def throttled_iter_content(chunk_size=1024, decode_unicode=False):
                start_time = time.time()
                bytes_downloaded = 0

                for chunk in original_iter_content(chunk_size=chunk_size, decode_unicode=decode_unicode):
                    if chunk:
                        bytes_downloaded += len(chunk)
                        yield chunk

                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            expected_time = bytes_downloaded / self.max_bytes_per_second
                            if expected_time > elapsed:
                                time.sleep(expected_time - elapsed)

            response.iter_content = throttled_iter_content

        return response


class BandwidthLimitedTransport(httpx.HTTPTransport):
    """HTTPX transport with bandwidth limiting"""

    def __init__(self, max_bytes_per_second: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.max_bytes_per_second = max_bytes_per_second
        self._start_time = None
        self._bytes_downloaded = 0

    def handle_request(self, request):
        self._start_time = time.time()
        self._bytes_downloaded = 0

        response = super().handle_request(request)

        content_length = response.headers.get('content-length')
        if (self.max_bytes_per_second and content_length and
            int(content_length) > 1024 * 1024):

            original_stream = response.stream

            def throttled_stream():
                for chunk in original_stream:
                    if chunk:
                        self._bytes_downloaded += len(chunk)
                        yield chunk

                        elapsed = time.time() - self._start_time
                        if elapsed > 0:
                            expected_time = self._bytes_downloaded / self.max_bytes_per_second
                            if expected_time > elapsed:
                                time.sleep(expected_time - elapsed)

            response.stream = throttled_stream()

        return response


def parse_hf_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse HuggingFace URL to extract repo_id and filename"""
    try:
        if 'huggingface.co' not in url:
            return None, None

        parts = url.split('/')
        if len(parts) < 7:
            return None, None

        repo_id = f"{parts[3]}/{parts[4]}"
        filename = '/'.join(parts[7:])

        return repo_id, filename

    except Exception:
        return None, None


def create_symlink(source_path: str, target_path: str, verbose: bool = False) -> tuple[bool, str]:
    """Create symlink from source to target.

    Args:
        source_path: Path to the source file (symlink target)
        target_path: Path where the symlink will be created
        verbose: If True, print debug info to console (used by queue processor)
    """
    try:
        target = Path(target_path).resolve()
        source = Path(source_path).resolve()

        if not source.exists():
            return False, f"Source file does not exist: {source_path}"

        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists() or target.is_symlink():
            if verbose:
                print(f"    Deleting existing: {target} (symlink={target.is_symlink()})")
            target.unlink()

        target.symlink_to(source)

        if target.is_symlink() and target.exists():
            if verbose:
                print(f"    Created symlink: {target} -> {source}")
            return True, "Symlink created successfully"
        else:
            if verbose:
                print(f"    VERIFY FAILED: is_symlink={target.is_symlink()}, exists={target.exists()}")
            return False, f"Symlink verification failed for {target_path}"

    except Exception as e:
        return False, f"Symlink creation failed: {str(e)}"


def find_in_hub(hub_db: Path, hub_models_dir: Path,
                url: str, sha256_hash: str = None,
                verbose: bool = False) -> Optional[str]:
    """Find model in HubRoot database by SHA256 hash or filename.

    HubRoot stores models at: {hub_models_dir}/{blake3_hash}/{filename}

    Priority:
    1. SHA256 hash lookup (most reliable - matches HuggingFace LFS OID)
    2. Filename match (fallback)

    Args:
        hub_db: Path to HubRoot's hubrootv3.db
        hub_models_dir: Path to HubRoot's models directory
        url: HuggingFace URL of the model
        sha256_hash: Optional SHA256 hash for lookup
        verbose: If True, print debug info to console
    """
    try:
        conn = sqlite3.connect(str(hub_db))
        cursor = conn.cursor()

        result = None

        # Priority 1: SHA256 hash lookup
        if sha256_hash:
            cursor.execute(
                "SELECT hash_blake3, filename FROM models WHERE hash_sha256 = ? AND deleted = 0",
                (sha256_hash,)
            )
            result = cursor.fetchone()
            if result and verbose:
                print(f"  Found via SHA256: {result[1]}")

        # Priority 2: Filename match
        if not result:
            filename = Path(urlparse(url).path).name
            cursor.execute(
                "SELECT hash_blake3, filename FROM models WHERE filename = ? AND deleted = 0",
                (filename,)
            )
            result = cursor.fetchone()
            if result and verbose:
                print(f"  Found via filename: {result[1]}")

        conn.close()

        if result:
            blake3_hash, hub_filename = result
            full_path = hub_models_dir / blake3_hash / hub_filename
            if full_path.exists():
                logging.info(f"Found model in hub: {hub_filename}")
                return str(full_path)

        return None

    except Exception as e:
        if verbose:
            print(f"  Error querying hub database: {e}")
        else:
            logging.error(f"Error querying hub database: {e}")
        return None


def load_config(config_file: str, create_if_missing: bool = True) -> dict:
    """Load configuration from YAML or JSON file.

    Args:
        config_file: Path to config file
        create_if_missing: If True, create default config file when missing
    """
    try:
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    return yaml.safe_load(f) or {}
                else:
                    return json.load(f)
        elif create_if_missing:
            with open(config_path, 'w') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
                else:
                    json.dump(DEFAULT_CONFIG, f, indent=2)
            return DEFAULT_CONFIG.copy()
        else:
            return DEFAULT_CONFIG.copy()
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return DEFAULT_CONFIG.copy()


def init_queue_table():
    """Initialize the download queue table if it doesn't exist"""
    conn = sqlite3.connect(QUEUE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS download_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            output_path TEXT NOT NULL,
            filename TEXT NOT NULL,
            config_name TEXT,
            remote_size INTEGER,
            hub_source_path TEXT,
            status TEXT DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            speed_mbps REAL DEFAULT 0,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')

    # Migration: add hub_source_path column if it doesn't exist
    cursor.execute("PRAGMA table_info(download_queue)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'hub_source_path' not in columns:
        cursor.execute("ALTER TABLE download_queue ADD COLUMN hub_source_path TEXT")

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_queue_status
        ON download_queue(status)
    ''')

    conn.commit()
    conn.close()
