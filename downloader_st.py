#!/usr/bin/env python3
"""
WanGP Smart Model Downloader - Streamlit UI Version
Clean web interface for downloading AI models from HuggingFace
"""

import os
import sys
import json
import yaml
import requests
import httpx
import time
import threading
import sqlite3
import logging
import subprocess
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

from hash_index import HashIndex

# Queue database path (shared with hfqueue.py)
QUEUE_DB_PATH = "hfcache.db"

try:
    from huggingface_hub import set_client_factory
except ImportError:
    try:
        from huggingface_hub import configure_http_backend
        set_client_factory = None
    except ImportError:
        set_client_factory = None
        configure_http_backend = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('downloader.log', mode='w'),
    ]
)


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


class ModelDownloader:
    def __init__(self, hub_dir: str = None, bandwidth_limit: Optional[int] = None, cache_dir: str = None, config_file: str = "config.yaml"):
        self.config = self.load_config(config_file)

        self.wan2gp_dir = Path(self.config.get("wan2gp_directory", "../Wan2GP-mryan"))
        self.cache_dir = Path(cache_dir) if cache_dir else self.wan2gp_dir / "ckpts"
        self.defaults_dir = self.wan2gp_dir / "defaults"
        self.bandwidth_limit = bandwidth_limit if bandwidth_limit is not None else self.config.get("bandwidth_limit_kb")

        # InvokeAI integration
        self.invokeai_db = Path(self.config.get("invokeai_db", "")) if self.config.get("invokeai_db") else None
        self.invokeai_models_dir = Path(self.config.get("invokeai_models_dir", "")) if self.config.get("invokeai_models_dir") else None
        self.invokeai_enabled = (
            self.invokeai_db is not None and
            self.invokeai_db.exists() and
            self.invokeai_models_dir is not None and
            self.invokeai_models_dir.exists()
        )

        # SHA256 hash index for InvokeAI models
        self.parallel_hash_workers = self.config.get("parallel_hash_workers", 8)
        self.hash_index = None
        if self.invokeai_enabled:
            self.hash_index = HashIndex(
                str(self.invokeai_db),
                str(self.invokeai_models_dir),
                self.parallel_hash_workers
            )

        # Legacy hub support
        self.hub_dir = Path(hub_dir) if hub_dir else None
        self.hub_enabled = self.hub_dir is not None and self.hub_dir.exists() and self.hub_dir.is_dir()

        self.hash_db = {}
        self.download_queue = []
        self.hf_api = HfApi()
        self.cache_db_path = "hfcache.db"
        self.init_cache_db()

        # Configure bandwidth-limited HTTP backend
        if self.bandwidth_limit:
            if set_client_factory is not None:
                def create_httpx_client():
                    transport = BandwidthLimitedTransport(
                        max_bytes_per_second=self.bandwidth_limit * 1024
                    )
                    return httpx.Client(transport=transport)
                set_client_factory(create_httpx_client)
            elif configure_http_backend is not None:
                def create_session():
                    return BandwidthLimitedSession(max_bytes_per_second=self.bandwidth_limit * 1024)
                configure_http_backend(backend_factory=create_session)

    def load_config(self, config_file: str) -> dict:
        """Load configuration from YAML or JSON file"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        return yaml.safe_load(f) or {}
                    else:
                        return json.load(f)
            else:
                default_config = self.create_default_config()
                with open(config_path, 'w') as f:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        yaml.dump(default_config, f, default_flow_style=False)
                    else:
                        json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return self.create_default_config()

    def create_default_config(self) -> dict:
        """Create default configuration"""
        return {
            "wan2gp_directory": "../Wan2GP-mryan",
            "bandwidth_limit_kb": 90000,
            "hub_directory": "",
        }

    def init_cache_db(self):
        """Initialize SQLite cache database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hf_file_cache (
                    cache_key TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_size INTEGER,
                    file_hash TEXT,
                    raw_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success INTEGER DEFAULT 1
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_repo_file
                ON hf_file_cache(repo_id, filename)
            ''')

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Could not initialize cache database: {e}")

    def get_cached_file_info(self, cache_key: str) -> dict:
        """Get cached file info from database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT file_size, file_hash, raw_data, success
                FROM hf_file_cache
                WHERE cache_key = ?
            ''', (cache_key,))

            result = cursor.fetchone()
            conn.close()

            if result:
                file_size, file_hash, raw_data, success = result
                if success:
                    return {
                        'found': True,
                        'data': {
                            'size': file_size,
                            'hash': file_hash,
                            'raw_data': raw_data
                        }
                    }
                else:
                    return {'found': True, 'data': None}

            return {'found': False, 'data': None}

        except Exception as e:
            logging.error(f"Error reading from cache: {e}")
            return {'found': False, 'data': None}

    def cache_file_info(self, cache_key: str, repo_id: str, filename: str, file_info=None, raw_data: str = None):
        """Cache file info to database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            if file_info:
                file_size = file_info.size if hasattr(file_info, 'size') else None
                file_hash = None
                if hasattr(file_info, 'lfs') and file_info.lfs:
                    if hasattr(file_info.lfs, 'oid'):
                        file_hash = file_info.lfs.oid
                    elif hasattr(file_info.lfs, 'sha256'):
                        file_hash = file_info.lfs.sha256
                success = 1
            else:
                file_size = None
                file_hash = None
                success = 0

            cursor.execute('''
                INSERT OR REPLACE INTO hf_file_cache
                (cache_key, repo_id, filename, file_size, file_hash, raw_data, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (cache_key, repo_id, filename, file_size, file_hash, raw_data, success))

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Error writing to cache: {e}")

    def clear_cache_db(self):
        """Clear all entries from cache database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM hf_file_cache")
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")

    def parse_hf_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
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

    def get_hf_file_info(self, url: str) -> Optional[dict]:
        """Get file info from HuggingFace API with caching"""
        repo_id, filename = self.parse_hf_url(url)
        if not repo_id or not filename:
            return None

        cache_key = f"{repo_id}/{filename}"

        cache_result = self.get_cached_file_info(cache_key)
        if cache_result['found']:
            return cache_result['data']

        try:
            file_info_list = self.hf_api.get_paths_info(repo_id, [filename])
            if file_info_list and len(file_info_list) > 0:
                info = file_info_list[0]

                file_hash = None
                if hasattr(info, 'lfs') and info.lfs:
                    if hasattr(info.lfs, 'oid'):
                        file_hash = info.lfs.oid
                    elif hasattr(info.lfs, 'sha256'):
                        file_hash = info.lfs.sha256

                self.cache_file_info(cache_key, repo_id, filename, info, json.dumps({'size': info.size}))

                return {
                    'size': info.size,
                    'hash': file_hash,
                }
        except Exception as e:
            logging.error(f"HF API call failed for {cache_key}: {e}")
            self.cache_file_info(cache_key, repo_id, filename, file_info=None, raw_data=str(e))

        return None

    def find_in_invokeai(self, url: str, sha256_hash: str = None) -> Optional[str]:
        """Find model in InvokeAI database by SHA256 hash, source URL, or filename.

        Priority:
        1. SHA256 hash lookup (most reliable, hash-based matching)
        2. Source URL exact match (legacy fallback)
        3. Filename pattern match (last resort)
        """
        if not self.invokeai_enabled:
            return None

        # Priority 1: SHA256 hash lookup via hash index
        if sha256_hash and self.hash_index and self.hash_index.is_ready():
            result = self.hash_index.lookup_by_sha256(sha256_hash)
            if result:
                file_path = Path(result['file_path'])
                if file_path.exists():
                    logging.info(f"Found model via SHA256: {file_path.name}")
                    return str(file_path)

        # Priority 2 & 3: URL and filename fallback
        try:
            conn = sqlite3.connect(str(self.invokeai_db))
            cursor = conn.cursor()

            # Try exact source URL match
            cursor.execute("SELECT path FROM models WHERE source = ?", (url,))
            result = cursor.fetchone()

            if not result:
                # Try filename pattern match
                filename = Path(urlparse(url).path).name
                cursor.execute("SELECT path FROM models WHERE source LIKE ?", (f"%/{filename}",))
                result = cursor.fetchone()

            conn.close()

            if result:
                relative_path = result[0]
                full_path = self.invokeai_models_dir / relative_path
                if full_path.exists():
                    return str(full_path)
            return None

        except Exception as e:
            logging.error(f"Error querying InvokeAI database: {e}")
            return None

    def create_symlink(self, source_path: str, target_path: str) -> tuple[bool, str]:
        """Create symlink from source to target"""
        try:
            target = Path(target_path).resolve()
            source = Path(source_path).resolve()

            if not source.exists():
                return False, f"Source file does not exist: {source_path}"

            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists() or target.is_symlink():
                target.unlink()

            target.symlink_to(source)

            if target.is_symlink() and target.exists():
                return True, "Symlink created successfully"
            else:
                return False, f"Symlink verification failed for {target_path}"

        except Exception as e:
            return False, f"Symlink creation failed: {str(e)}"

    def determine_output_path(self, url: str, url_type: str) -> str:
        """Determine the correct output path based on file type and URL"""
        filename = Path(urlparse(url).path).name
        base = str(self.wan2gp_dir)

        if any(pattern in filename.lower() for pattern in ['lora', 'adapter']):
            if 'flux' in filename.lower():
                return f"{base}/loras_flux/{filename}"
            elif 'hunyuan' in filename.lower():
                return f"{base}/loras_hunyuan/{filename}"
            elif 'i2v' in filename.lower():
                return f"{base}/loras_i2v/{filename}"
            elif 'ltxv' in filename.lower() or 'ltx' in filename.lower():
                return f"{base}/loras_ltxv/{filename}"
            elif 'qwen' in filename.lower():
                return f"{base}/loras_qwen/{filename}"
            else:
                return f"{base}/loras/{filename}"
        else:
            return f"{base}/ckpts/{filename}"

    def resolve_config_urls(self, config_file: str) -> List[Tuple[str, str, str]]:
        """Resolve URLs from config file"""
        defaults_dir = self.defaults_dir

        def resolve_urls(urls):
            if isinstance(urls, str):
                ref_file = defaults_dir / f"{urls}.json"
                if ref_file.exists():
                    try:
                        with open(ref_file, 'r') as f:
                            ref_config = json.load(f)
                        ref_urls = ref_config.get('model', {}).get('URLs', [])
                        return resolve_urls(ref_urls)
                    except:
                        return []
                return []
            elif isinstance(urls, list):
                return [url for url in urls if isinstance(url, str) and url.startswith('http')]
            else:
                return []

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            model_data = config.get('model', {})
            raw_urls = model_data.get('URLs', [])
            preload_urls = model_data.get('preload_URLs', [])

            resolved_urls = resolve_urls(raw_urls)
            resolved_preload_urls = resolve_urls(preload_urls) if isinstance(preload_urls, list) else []

            results = []

            for url in resolved_urls:
                if url.startswith('http'):
                    output_path = self.determine_output_path(url, "MAIN")
                    results.append(("MAIN", url, output_path))

            for url in resolved_preload_urls:
                if url.startswith('http'):
                    output_path = self.determine_output_path(url, "PRELOAD")
                    results.append(("PRELOAD", url, output_path))

            return results

        except Exception:
            return []

    def get_batch_file_sizes(self, cache_keys: List[str]) -> Dict[str, int]:
        """Get multiple file sizes from cache in a single query"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            placeholders = ','.join('?' * len(cache_keys))

            cursor.execute(f'''
                SELECT cache_key, file_size
                FROM hf_file_cache
                WHERE cache_key IN ({placeholders}) AND success = 1
            ''', cache_keys)

            results = cursor.fetchall()
            conn.close()

            return {cache_key: file_size for cache_key, file_size in results}

        except Exception as e:
            logging.error(f"Error reading batch from cache: {e}")
            return {}

    def build_download_queue(self, progress_callback=None) -> List[Dict]:
        """Build complete download queue from all config files"""
        queue = []

        if not self.defaults_dir.exists():
            logging.error(f"Defaults directory not found: {self.defaults_dir}")
            return queue

        # First pass: collect all URLs
        url_to_cache_key = {}
        for config_file in sorted(self.defaults_dir.glob("*.json")):
            config_name = config_file.stem
            urls = self.resolve_config_urls(str(config_file))

            for url_type, url, output_path in urls:
                if url.startswith('http'):
                    repo_id, filename = self.parse_hf_url(url)
                    if repo_id and filename:
                        cache_key = f"{repo_id}/{filename}"
                        url_to_cache_key[url] = cache_key

                queue.append({
                    'config': config_name,
                    'type': url_type,
                    'url': url,
                    'output_path': output_path,
                    'filename': Path(output_path).name,
                    'status': '---',
                    'progress': 0,
                    'remote_size': None
                })

        # Second pass: batch lookup file sizes
        if url_to_cache_key:
            cache_keys = list(url_to_cache_key.values())
            cached_sizes = self.get_batch_file_sizes(cache_keys)

            cache_misses = []
            for item in queue:
                url = item['url']
                if url in url_to_cache_key:
                    cache_key = url_to_cache_key[url]
                    if cache_key in cached_sizes:
                        item['remote_size'] = cached_sizes[cache_key]
                    else:
                        cache_misses.append((item, cache_key, url))

            # Fetch missing sizes
            if cache_misses and progress_callback:
                total_misses = len(cache_misses)

                repo_batches = {}
                for item, cache_key, url in cache_misses:
                    repo_id, filename = self.parse_hf_url(url)
                    if repo_id and filename:
                        if repo_id not in repo_batches:
                            repo_batches[repo_id] = []
                        repo_batches[repo_id].append((item, cache_key, filename, url))

                processed = 0
                for repo_id, batch_items in repo_batches.items():
                    try:
                        filenames = [filename for _, _, filename, _ in batch_items]
                        file_info_list = self.hf_api.get_paths_info(repo_id, filenames)

                        info_by_filename = {info.path: info for info in file_info_list if hasattr(info, 'path')}

                        for item, cache_key, filename, url in batch_items:
                            processed += 1
                            progress_callback(processed, total_misses)

                            if filename in info_by_filename:
                                info = info_by_filename[filename]
                                if hasattr(info, 'size'):
                                    self.cache_file_info(cache_key, repo_id, filename, info, json.dumps({'size': info.size}))
                                    item['remote_size'] = info.size

                    except Exception as e:
                        for item, cache_key, filename, url in batch_items:
                            processed += 1
                            progress_callback(processed, total_misses)
                            self.cache_file_info(cache_key, repo_id, filename, None, str(e))

        # Third pass: check InvokeAI and create symlinks
        if self.invokeai_enabled:
            # Get all SHA256 hashes from cache in one query
            all_cache_keys = list(url_to_cache_key.values()) if url_to_cache_key else []
            sha256_by_key = {}
            if all_cache_keys:
                try:
                    conn = sqlite3.connect(self.cache_db_path)
                    cursor = conn.cursor()
                    placeholders = ','.join('?' * len(all_cache_keys))
                    cursor.execute(f'''
                        SELECT cache_key, file_hash
                        FROM hf_file_cache
                        WHERE cache_key IN ({placeholders}) AND file_hash IS NOT NULL
                    ''', all_cache_keys)
                    sha256_by_key = dict(cursor.fetchall())
                    conn.close()
                except Exception as e:
                    logging.error(f"Error fetching SHA256 hashes: {e}")

            for item in queue:
                output_path = Path(item['output_path'])

                if output_path.exists():
                    if output_path.is_symlink():
                        item['status'] = 'Linked'
                    else:
                        item['status'] = 'Exists'
                    continue

                if not item['url'].startswith('http'):
                    continue

                # Get SHA256 hash for this item
                cache_key = url_to_cache_key.get(item['url'])
                sha256_hash = sha256_by_key.get(cache_key) if cache_key else None

                invokeai_path = self.find_in_invokeai(item['url'], sha256_hash=sha256_hash)
                if invokeai_path:
                    success, msg = self.create_symlink(invokeai_path, str(output_path))
                    if success:
                        item['status'] = 'Linked'

        return queue

    def download_hf_file(self, url: str, output_path: str, item: Dict, progress_callback=None) -> tuple[bool, str]:
        """Download file from HuggingFace"""
        try:
            repo_id, filename = self.parse_hf_url(url)
            if not repo_id or not filename:
                return False, f"Failed to parse URL: {url}"

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            expected_size = item.get('remote_size', 0)

            download_result = {'cached_file': None, 'error': None}
            download_done = threading.Event()

            def do_download():
                try:
                    download_result['cached_file'] = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=str(self.cache_dir)
                    )
                except Exception as e:
                    download_result['error'] = str(e)
                finally:
                    download_done.set()

            download_thread = threading.Thread(target=do_download)
            download_thread.start()

            # Poll progress
            if progress_callback and expected_size > 0:
                start_time = time.time()
                cache_path = Path(self.cache_dir)
                last_size = 0

                while not download_done.is_set():
                    incomplete_files = list(cache_path.rglob("*.incomplete"))

                    if incomplete_files:
                        incomplete_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        active_file = incomplete_files[0]

                        try:
                            current_size = active_file.stat().st_size

                            if current_size > last_size:
                                last_size = current_size
                                elapsed = time.time() - start_time
                                speed = current_size / elapsed if elapsed > 0 else 0
                                speed_mb = speed / (1024 * 1024)
                                percent = (current_size / expected_size) * 100 if expected_size > 0 else 0
                                progress_callback(current_size, expected_size, min(percent, 99.9), speed_mb)
                        except (FileNotFoundError, OSError):
                            pass

                    download_done.wait(timeout=0.5)

            download_thread.join()

            if download_result['error']:
                return False, f"Download error: {download_result['error']}"

            cached_file = download_result['cached_file']

            if progress_callback and expected_size > 0:
                progress_callback(expected_size, expected_size, 100.0, 0)

            if not Path(cached_file).exists():
                return False, f"HuggingFace returned non-existent file: {cached_file}"

            success, msg = self.create_symlink(cached_file, output_path)
            if success:
                return True, "Success"
            else:
                return False, msg

        except Exception as e:
            return False, f"Download error: {str(e)}"


# Queue Management Functions
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
            status TEXT DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            speed_mbps REAL DEFAULT 0,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_queue_status
        ON download_queue(status)
    ''')

    conn.commit()
    conn.close()


def add_to_queue(items: List[Dict]) -> int:
    """Add items to the download queue"""
    init_queue_table()
    conn = sqlite3.connect(QUEUE_DB_PATH)
    cursor = conn.cursor()

    added = 0
    for item in items:
        # Check if already in queue (pending or downloading)
        cursor.execute('''
            SELECT id FROM download_queue
            WHERE output_path = ? AND status IN ('pending', 'downloading')
        ''', (item['output_path'],))

        if cursor.fetchone():
            continue  # Skip if already queued

        cursor.execute('''
            INSERT INTO download_queue (url, output_path, filename, config_name, remote_size, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
        ''', (item['url'], item['output_path'], item['filename'], item.get('config', ''), item.get('remote_size')))
        added += 1

    conn.commit()
    conn.close()
    return added


def get_queue_items() -> List[Dict]:
    """Get all items from the download queue"""
    init_queue_table()
    conn = sqlite3.connect(QUEUE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, url, output_path, filename, config_name, remote_size,
               status, progress, speed_mbps, error_message, created_at, started_at, completed_at
        FROM download_queue
        ORDER BY created_at DESC
    ''')

    rows = cursor.fetchall()
    conn.close()

    items = []
    for row in rows:
        items.append({
            'id': row[0],
            'url': row[1],
            'output_path': row[2],
            'filename': row[3],
            'config_name': row[4],
            'remote_size': row[5],
            'status': row[6],
            'progress': row[7],
            'speed_mbps': row[8],
            'error_message': row[9],
            'created_at': row[10],
            'started_at': row[11],
            'completed_at': row[12]
        })

    return items


def get_queue_stats() -> Dict:
    """Get queue statistics"""
    init_queue_table()
    conn = sqlite3.connect(QUEUE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT status, COUNT(*)
        FROM download_queue
        GROUP BY status
    ''')

    stats = dict(cursor.fetchall())
    conn.close()

    # Combine 'complete' and 'linked' as both are successful completions
    complete_count = stats.get('complete', 0) + stats.get('linked', 0)
    return {
        'pending': stats.get('pending', 0),
        'downloading': stats.get('downloading', 0),
        'complete': complete_count,
        'failed': stats.get('failed', 0)
    }


def clear_queue(status_filter: str = None):
    """Clear queue entries by status"""
    conn = sqlite3.connect(QUEUE_DB_PATH)
    cursor = conn.cursor()

    if status_filter:
        # 'complete' filter should also clear 'linked' entries (both are successful)
        if status_filter == 'complete':
            cursor.execute("DELETE FROM download_queue WHERE status IN ('complete', 'linked')")
        else:
            cursor.execute("DELETE FROM download_queue WHERE status = ?", (status_filter,))
    else:
        cursor.execute("DELETE FROM download_queue")

    conn.commit()
    conn.close()


def remove_queue_item(item_id: int):
    """Remove a specific item from the queue"""
    conn = sqlite3.connect(QUEUE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM download_queue WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()


def is_queue_processor_running() -> bool:
    """Check if hfqueue processor is running"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'hfqueue.py'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def start_queue_processor() -> bool:
    """Start the queue processor as a background process"""
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.absolute()
        hfqueue_path = script_dir / "hfqueue.py"

        # Start as detached process
        subprocess.Popen(
            ['python', str(hfqueue_path)],
            cwd=str(script_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # Detach from parent process
        )
        return True
    except Exception as e:
        logging.error(f"Failed to start queue processor: {e}")
        return False


def stop_queue_processor() -> bool:
    """Stop the queue processor"""
    try:
        result = subprocess.run(
            ['pkill', '-f', 'hfqueue.py'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


# Streamlit UI
def format_file_size(size_bytes: Optional[int]) -> str:
    """Format file size in bytes to human readable GB"""
    if size_bytes is None:
        return "Unknown"

    gb = size_bytes / (1024 ** 3)
    if gb < 0.1:
        return f"{gb:.2f}"
    elif gb < 1:
        return f"{gb:.1f}"
    else:
        return f"{gb:.1f}"


def get_item_source(downloader, item) -> str:
    """Determine source for an item"""
    output_path = Path(item['output_path'])

    if output_path.exists() and not output_path.is_symlink():
        return "ckpt"

    if output_path.exists() and output_path.is_symlink():
        return "link"

    if downloader.invokeai_enabled and item['url'].startswith('http'):
        invokeai_path = downloader.find_in_invokeai(item['url'])
        if invokeai_path:
            return "link"

    return "download"


def is_high_precision(filename: str) -> bool:
    """Check if filename indicates high precision (non-quantized)"""
    name = filename.lower()
    # Quantized files have 'quanto' in the name
    if '_quanto_' in name:
        return False
    # High precision files typically end with bf16, fp16, mbf16, mfp16
    if any(name.endswith(f'_{p}.safetensors') for p in ['bf16', 'fp16', 'mbf16', 'mfp16']):
        return True
    # Files without precision suffix are also considered "high" (full model)
    return True


def group_high_low_models(queue_items: List[Dict]) -> List[Dict]:
    """
    Group HIGH/LOW model pairs into single entries.
    Only pairs configs that have exactly 2 files (1 high + 1 low).
    Configs with 3+ files are shown as individual entries (alternatives).
    """
    from collections import defaultdict

    # Group by config name
    configs = defaultdict(list)
    for item in queue_items:
        configs[item['config']].append(item)

    result = []
    for config_name, items in configs.items():
        # Only pair if exactly 2 files: 1 high-precision + 1 quantized
        if len(items) == 2:
            high_items = [i for i in items if is_high_precision(i['filename'])]
            low_items = [i for i in items if not is_high_precision(i['filename'])]

            if len(high_items) == 1 and len(low_items) == 1:
                # Valid HIGH/LOW pair
                high_item = high_items[0]
                low_item = low_items[0]

                merged = high_item.copy()
                merged['_is_pair'] = True
                merged['_high_item'] = high_item
                merged['_low_items'] = [low_item]
                merged['_display_name'] = f"{high_item['filename'].replace('.safetensors', '')} (high/low)"

                # Combined status
                statuses = [high_item['status'], low_item['status']]
                if '---' in statuses:
                    merged['status'] = '---'
                elif all(s in ['Exists', 'Linked', 'Complete'] for s in statuses):
                    merged['status'] = 'Complete'
                else:
                    merged['status'] = high_item['status']

                # Combined size
                total_size = (high_item.get('remote_size') or 0) + (low_item.get('remote_size') or 0)
                merged['_total_size'] = total_size if total_size > 0 else None

                result.append(merged)
                continue

        # Not a pair - add items individually
        for item in items:
            item['_is_pair'] = False
            item['_display_name'] = item['filename']
            result.append(item)

    return result


def update_model_statuses():
    """Check filesystem and update model statuses"""
    changed = False
    for item in st.session_state.download_queue:
        output_path = Path(item['output_path'])
        if output_path.exists() and item['status'] == '---':
            if output_path.is_symlink():
                item['status'] = 'Linked'
            else:
                item['status'] = 'Exists'
            changed = True
    return changed


@st.fragment(run_every="10s")
def models_status_checker():
    """Background checker that triggers refresh when models are downloaded"""
    if update_model_statuses():
        st.rerun()


def render_models_tab():
    """Render the Models tab content"""
    downloader = st.session_state.downloader
    show_all = st.session_state.get('show_all', False)
    filter_text = st.session_state.get('filter_input', '')

    if not st.session_state.download_queue:
        st.info("No models found in download queue. Check your config.yaml settings.")
        return

    # Background status checker (runs every 10s)
    models_status_checker()

    # Update status for all items first
    update_model_statuses()

    # Group HIGH/LOW pairs
    grouped_queue = group_high_low_models(st.session_state.download_queue)

    # Filter grouped queue
    filtered_queue = []
    for item in grouped_queue:
        # Filter by text
        display_name = item.get('_display_name', item['filename'])
        if filter_text and filter_text.lower() not in display_name.lower():
            continue

        # Filter by show_all
        if not show_all and item['status'] in ['Exists', 'Linked', 'Complete']:
            continue

        filtered_queue.append(item)

    # Convert to DataFrame
    df_data = []
    for item in filtered_queue:
        wan2gp_prefix = str(downloader.wan2gp_dir) + "/"
        destination = item['output_path'].replace(wan2gp_prefix, "").split("/")[0]

        # Use combined size for pairs, otherwise individual size
        if item.get('_is_pair') and item.get('_total_size'):
            file_size = item['_total_size']
        elif Path(item['output_path']).exists():
            file_size = Path(item['output_path']).stat().st_size
        else:
            file_size = item.get('remote_size')

        # Use display name (includes "(high/low)" for pairs)
        display_name = item.get('_display_name', item['filename'])

        df_data.append({
            'Config': item['config'],
            'Model File': display_name,
            'Size (GB)': format_file_size(file_size),
            'Source': get_item_source(downloader, item),
            'Status': item['status'],
            'Dest': destination,
            '_output_path': item['output_path'],  # Used as unique key
            '_is_pair': item.get('_is_pair', False),
            '_item_ref': item,  # Reference to full item for queue addition
        })

    df = pd.DataFrame(df_data)

    # Handle selection BEFORE rendering buttons
    if 'model_table' in st.session_state:
        if 'selection' in st.session_state.model_table and 'rows' in st.session_state.model_table.selection:
            selected_rows = st.session_state.model_table.selection['rows']
            st.session_state.selected_items = set(df.iloc[selected_rows]['_output_path'].tolist())

    # Compact action bar
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 2])

    with col1:
        if st.button("Select All", width="stretch", key="select_all"):
            st.session_state.selected_items = set(df['_output_path'].tolist())
            st.rerun()

    with col2:
        if st.button("Clear", width="stretch", key="clear_all"):
            st.session_state.selected_items = set()
            st.rerun()

    with col3:
        selected_count = len(st.session_state.selected_items)
        if st.button(f"Add to Queue ({selected_count})", width="stretch", disabled=selected_count == 0, type="primary"):
            # Build list of items to add to queue
            # For pairs, add both HIGH and LOW items
            queue_items = []
            for _, row in df.iterrows():
                if row['_output_path'] not in st.session_state.selected_items:
                    continue

                item = row['_item_ref']
                if item.get('_is_pair'):
                    # Add HIGH item
                    high = item['_high_item']
                    queue_items.append({
                        'url': high['url'],
                        'output_path': high['output_path'],
                        'filename': high['filename'],
                        'config': high['config'],
                        'remote_size': high.get('remote_size')
                    })
                    # Add all LOW items
                    for low in item['_low_items']:
                        queue_items.append({
                            'url': low['url'],
                            'output_path': low['output_path'],
                            'filename': low['filename'],
                            'config': low['config'],
                            'remote_size': low.get('remote_size')
                        })
                else:
                    # Single item
                    queue_items.append({
                        'url': item['url'],
                        'output_path': item['output_path'],
                        'filename': item['filename'],
                        'config': item['config'],
                        'remote_size': item.get('remote_size')
                    })

            added = add_to_queue(queue_items)
            st.session_state.selected_items = set()
            st.toast(f"Added {added} items to download queue")
            st.rerun()

    with col4:
        st.write("")

    with col5:
        st.markdown(f"**{selected_count}** of **{len(filtered_queue)}** selected")
        view_mode = "Showing All" if show_all else "Showing Not Downloaded"
        st.caption(view_mode)

    # Display table - hide internal columns
    display_columns = ['Config', 'Model File', 'Size (GB)', 'Source', 'Status', 'Dest']
    st.dataframe(
        df[display_columns],
        width="stretch",
        selection_mode="multi-row",
        on_select="rerun",
        key="model_table",
        hide_index=True
    )


@st.fragment(run_every="3s")
def render_queue_status():
    """Auto-refreshing queue status display"""
    stats = get_queue_stats()
    processor_running = is_queue_processor_running()

    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 2])

    with col1:
        st.metric("Pending", stats['pending'])

    with col2:
        st.metric("Downloading", stats['downloading'])

    with col3:
        st.metric("Complete", stats['complete'])

    with col4:
        st.metric("Failed", stats['failed'])

    with col5:
        if processor_running:
            st.success("Running")
        else:
            st.warning("Stopped")

    with col6:
        # Start/Stop processor buttons
        if processor_running:
            if st.button("Stop Processor", width="stretch", type="secondary", key="stop_processor"):
                if stop_queue_processor():
                    st.toast("Processor stopped")
                    time.sleep(0.5)
                else:
                    st.error("Failed to stop processor")
        else:
            if st.button("Start Processor", width="stretch", type="primary", key="start_processor"):
                if start_queue_processor():
                    st.toast("Processor started")
                    time.sleep(0.5)
                else:
                    st.error("Failed to start processor")

    st.divider()

    # Queue items table
    queue_items = get_queue_items()

    if not queue_items:
        st.info("Download queue is empty. Select models in the Models tab and click 'Add to Queue'.")
        return

    # Build dataframe
    df_data = []
    for item in queue_items:
        status_icon = {
            'pending': '⏳',
            'downloading': '📥',
            'complete': '✅',
            'failed': '❌',
            'linked': '🔗'
        }.get(item['status'], '❓')

        progress_str = f"{item['progress']}%" if item['progress'] else "-"
        speed_str = f"{item['speed_mbps']:.1f} MB/s" if item['speed_mbps'] else "-"
        size_str = format_file_size(item['remote_size']) if item['remote_size'] else "-"

        df_data.append({
            'ID': item['id'],
            'Status': f"{status_icon} {item['status']}",
            'Filename': item['filename'],
            'Size (GB)': size_str,
            'Progress': progress_str,
            'Speed': speed_str,
            'Error': item['error_message'] or "-",
            'Added': item['created_at'][:16] if item['created_at'] else "-"
        })

    df = pd.DataFrame(df_data)

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            'ID': st.column_config.NumberColumn(width="small"),
            'Status': st.column_config.TextColumn(width="medium"),
            'Filename': st.column_config.TextColumn(width="large"),
            'Size (GB)': st.column_config.TextColumn(width="small"),
            'Progress': st.column_config.TextColumn(width="small"),
            'Speed': st.column_config.TextColumn(width="small"),
            'Error': st.column_config.TextColumn(width="medium"),
            'Added': st.column_config.TextColumn(width="medium"),
        }
    )


def compute_file_sha256(file_path: Path) -> Optional[str]:
    """Compute SHA256 hash of a local file"""
    if not file_path.exists() or file_path.is_dir():
        return None
    try:
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8 * 1024 * 1024):  # 8MB chunks
                sha256.update(chunk)
        return sha256.hexdigest()
    except (IOError, OSError):
        return None


def queue_hub_models() -> Tuple[int, int]:
    """
    Queue all models that exist in the InvokeAI hub.
    For each model found in hub:
    1. Delete existing file in Wan2GP destination (symlink or full file)
    2. Add to download queue (will be re-symlinked by processor)

    Uses multiple matching strategies:
    1. SHA256 hash from HuggingFace metadata
    2. URL matching in InvokeAI database
    3. For physical files: compute actual SHA256 and match against hub

    Returns (queued_count, deleted_count)
    """
    downloader = st.session_state.downloader

    if not downloader.invokeai_enabled:
        return 0, 0

    queued_count = 0
    deleted_count = 0
    queue_items = []

    for item in st.session_state.download_queue:
        if not item['url'].startswith('http'):
            continue

        output_path = Path(item['output_path'])
        invokeai_path = None

        # Strategy 1: Try URL/HF-hash matching first
        repo_id, filename = downloader.parse_hf_url(item['url'])
        sha256_hash = None
        if repo_id and filename:
            cache_key = f"{repo_id}/{filename}"
            cache_result = downloader.get_cached_file_info(cache_key)
            if cache_result['found'] and cache_result['data']:
                sha256_hash = cache_result['data'].get('hash')

        invokeai_path = downloader.find_in_invokeai(item['url'], sha256_hash=sha256_hash)

        # Strategy 2: If no match and physical file exists, compute its hash and lookup
        if not invokeai_path and output_path.exists() and not output_path.is_symlink():
            # Physical file exists - compute its actual SHA256
            file_hash = compute_file_sha256(output_path)
            if file_hash and downloader.hash_index:
                result = downloader.hash_index.lookup_by_sha256(file_hash)
                if result:
                    invokeai_path = result.get('file_path')
                    if invokeai_path and Path(invokeai_path).exists():
                        logging.info(f"Matched {output_path.name} via content hash")
                    else:
                        invokeai_path = None

        if invokeai_path:
            # Delete existing file if it exists (symlink or full file)
            if output_path.exists() or output_path.is_symlink():
                try:
                    output_path.unlink()
                    deleted_count += 1
                    # Reset status since file is gone
                    item['status'] = '---'
                except Exception as e:
                    logging.error(f"Failed to delete {output_path}: {e}")

            # Add to queue
            queue_items.append({
                'url': item['url'],
                'output_path': item['output_path'],
                'filename': item['filename'],
                'config': item.get('config', ''),
                'remote_size': item.get('remote_size')
            })

    if queue_items:
        queued_count = add_to_queue(queue_items)

    return queued_count, deleted_count


def render_queue_tab():
    """Render the Queue tab content"""
    # Action buttons (outside fragment so they don't auto-refresh)
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])

    with col1:
        if st.button("Queue Hub", width="stretch", type="primary", key="queue_hub"):
            with st.spinner("Finding hub models (computing hashes for physical files)..."):
                queued, deleted = queue_hub_models()
            st.toast(f"Queued {queued} hub models, deleted {deleted} existing files")

    with col2:
        if st.button("Clear Complete", width="stretch", key="clear_complete"):
            clear_queue('complete')
            st.toast("Cleared completed items")

    with col3:
        if st.button("Clear Failed", width="stretch", key="clear_failed"):
            clear_queue('failed')
            st.toast("Cleared failed items")

    with col4:
        if st.button("Clear All", width="stretch", key="clear_all_queue"):
            clear_queue()
            st.toast("Cleared all queue items")

    with col5:
        st.caption("Auto-refresh: 3s")

    with col6:
        st.write("")

    st.divider()

    # Auto-refreshing status and table
    render_queue_status()


def main():
    st.set_page_config(
        page_title="WanGP Model Downloader",
        page_icon="⬇️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS for responsive dataframe height
    st.markdown("""
        <style>
        /* Make dataframe fill available vertical space */
        .stDataFrame {
            height: calc(100vh - 280px) !important;
        }
        .stDataFrame > div {
            height: 100% !important;
        }
        .stDataFrame iframe {
            height: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize queue table
    init_queue_table()

    # Initialize session state
    if 'downloader' not in st.session_state:
        st.session_state.downloader = ModelDownloader(config_file="config.yaml")
        st.session_state.hash_index_ready = False
        st.session_state.selected_items = set()
        st.session_state.download_queue = []

    downloader = st.session_state.downloader

    # Initialize hash index if InvokeAI is enabled
    if downloader.hash_index and not st.session_state.get('hash_index_ready', False):
        hash_index = downloader.hash_index

        # Sync from InvokeAI database
        with st.spinner("Syncing hash index from InvokeAI..."):
            added = hash_index.sync_from_invokeai()

        stats = hash_index.get_stats()

        if stats['pending'] > 0:
            # Show progress UI for hash computation
            st.info(f"Computing SHA256 hashes for {stats['pending']} models. This is required for accurate model matching.")
            progress_bar = st.progress(0, text="Computing SHA256 hashes...")
            status_text = st.empty()

            def hash_progress_callback(current, total, filename):
                pct = current / total if total > 0 else 0
                progress_bar.progress(pct, text=f"Hashing {current}/{total}")
                status_text.caption(f"Current: {filename[:60]}...")

            # Compute hashes
            hash_index.compute_pending_hashes(progress_callback=hash_progress_callback)

            progress_bar.empty()
            status_text.empty()
            st.success(f"Hash index ready! {stats['total']} models indexed.")

        st.session_state.hash_index_ready = True

    # Build download queue after hash index is ready
    if not st.session_state.download_queue:
        with st.spinner("Building download queue..."):
            st.session_state.download_queue = downloader.build_download_queue()

    # Sidebar
    with st.sidebar:
        st.markdown("### WanGP Model Downloader")

        if st.button("🔄 Refresh Models", width="stretch", type="primary"):
            with st.spinner("Building download queue..."):
                progress_bar = st.progress(0, text="Building HuggingFace cache...")

                def progress_callback(current, total):
                    progress_bar.progress(current / total, text=f"Fetching metadata... {current}/{total}")

                st.session_state.download_queue = downloader.build_download_queue(progress_callback=progress_callback)
                progress_bar.empty()

            st.success(f"Loaded {len(st.session_state.download_queue)} models")

        st.divider()

        # Bandwidth limit (shown in sidebar for reference)
        st.markdown("**Bandwidth Limit**")
        current_limit_kb = downloader.bandwidth_limit if downloader.bandwidth_limit else 90000
        current_limit_mb = current_limit_kb / 1024
        st.caption(f"Current: {current_limit_mb:.0f} MB/s (~{current_limit_mb * 8:.0f} Mbps)")
        st.caption("Edit config.yaml to change")

        st.divider()

        # Filter
        filter_text = st.text_input("🔍 Filter models", placeholder="Type to search...", key="filter_input")

        show_all = st.checkbox("Show all files", value=False, key="show_all")

        st.divider()

        if st.button("🗑️ Clear HF Cache", width="stretch"):
            with st.spinner("Clearing cache and rebuilding..."):
                downloader.clear_cache_db()
                progress_bar = st.progress(0, text="Fetching fresh metadata...")

                def progress_callback(current, total):
                    progress_bar.progress(current / total, text=f"Fetching metadata... {current}/{total}")

                st.session_state.download_queue = downloader.build_download_queue(progress_callback=progress_callback)
                progress_bar.empty()

            st.success("Cache cleared!")
            st.rerun()

        # Hash index status
        if downloader.hash_index:
            st.divider()
            st.markdown("**SHA256 Hash Index**")
            hash_stats = downloader.hash_index.get_stats()
            if hash_stats['exists']:
                st.caption(f"Models: {hash_stats['total']} | Pending: {hash_stats['pending']}")
                if hash_stats['pending'] > 0:
                    st.warning(f"{hash_stats['pending']} hashes pending")
            else:
                st.caption("Not initialized")

            if st.button("🔄 Rebuild Hash Index", width="stretch"):
                from hash_index import rebuild_index
                with st.spinner("Rebuilding hash index..."):
                    progress_bar = st.progress(0, text="Computing SHA256 hashes...")

                    def hash_rebuild_callback(current, total, filename):
                        pct = current / total if total > 0 else 0
                        progress_bar.progress(pct, text=f"Hashing {current}/{total}")

                    rebuild_index(
                        str(downloader.invokeai_db),
                        str(downloader.invokeai_models_dir),
                        downloader.parallel_hash_workers,
                        hash_rebuild_callback
                    )
                    progress_bar.empty()

                st.success("Hash index rebuilt!")
                st.session_state.hash_index_ready = True
                st.rerun()

        # Queue stats at bottom
        stats = get_queue_stats()
        if stats['pending'] > 0 or stats['downloading'] > 0:
            st.divider()
            st.markdown("**Queue Status**")
            st.caption(f"⏳ {stats['pending']} pending | 📥 {stats['downloading']} active")

    # Main content with tabs
    tab1, tab2 = st.tabs(["📦 Models", "📥 Queue"])

    with tab1:
        render_models_tab()

    with tab2:
        render_queue_tab()


if __name__ == "__main__":
    main()
