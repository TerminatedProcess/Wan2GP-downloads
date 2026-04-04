#!/usr/bin/env python3
"""
Shared utilities for WanGP Model Downloader.
Used by both downloader_st.py (Streamlit UI) and hfqueue.py (queue processor).
"""

import ast
import json
import yaml
import time
import sqlite3
import logging
import posixpath
from pathlib import Path
from typing import Optional, Tuple, Dict, List
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


def save_config_value(config_file: str, key: str, value):
    """Update a single key in the config file, preserving comments and other keys."""
    config = load_config(config_file, create_if_missing=False)
    config[key] = value
    try:
        config_path = Path(config_file)
        with open(config_path, 'r') as f:
            lines = f.readlines()

        # Try to update the key in-place to preserve comments
        updated = False
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(f'{key}:'):
                indent = line[:len(line) - len(stripped)]
                lines[i] = f'{indent}{key}: {value}\n'
                updated = True
                break

        if updated:
            with open(config_path, 'w') as f:
                f.writelines(lines)
        else:
            # Key doesn't exist yet, append it
            with open(config_path, 'a') as f:
                f.write(f'\n{key}: {value}\n')

    except Exception as e:
        logging.error(f"Error saving config value: {e}")


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

    # Supplemental models table - for models not in defaults/*.json
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS supplemental_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_name TEXT NOT NULL,
            url TEXT NOT NULL,
            url_type TEXT DEFAULT 'MAIN',
            text_encoder_folder TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(url)
        )
    ''')

    conn.commit()
    conn.close()


def _extract_urls_from_list(list_node: ast.List, local_vars: Dict[str, str]) -> List[Tuple[str, str]]:
    """Extract (url, text_encoder_folder) tuples from an AST List of build_hf_url() calls."""
    results = []
    for elt in list_node.elts:
        if not isinstance(elt, ast.Call):
            continue
        if not (isinstance(elt.func, ast.Name) and elt.func.id == "build_hf_url"):
            continue

        args = []
        for arg in elt.args:
            val = _resolve_ast_value(arg, local_vars)
            if val is None:
                break
            args.append(val)
        else:
            if len(args) >= 2:
                repo_id = args[0]
                path = posixpath.join(*args[1:])
                url = f"https://huggingface.co/{repo_id}/resolve/main/{path}"
                te_folder = args[1] if len(args) >= 3 else ""
                results.append((url, te_folder))
    return results


def scan_handler_text_encoders(wan2gp_dir: Path) -> List[Dict]:
    """Scan Wan2GP handler .py files for text_encoder_URLs built via build_hf_url().

    Uses AST parsing to extract build_hf_url() calls from:
    1. Direct assignments: extra_model_def["text_encoder_URLs"] = [...]
    2. Dict literals: { "text_encoder_URLs": [...], ... }

    Resolves variable references (text_encoder_folder, text_encoder_repo) from nearby
    string assignments in the same function scope.

    Returns list of dicts with keys: config_name, url, text_encoder_folder
    """
    models_dir = wan2gp_dir / "models"
    if not models_dir.exists():
        return []

    results = []
    handler_files = list(models_dir.rglob("*_handler.py"))

    for handler_file in handler_files:
        try:
            source = handler_file.read_text()
            tree = ast.parse(source)
        except Exception:
            continue

        config_name = handler_file.stem.replace("_handler", "")

        # Also collect module-level string constants for variable resolution
        # Two passes: first plain strings, then f-strings that reference them
        module_vars = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and hasattr(node, 'lineno'):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        module_vars[target.id] = node.value.value
        # Second pass: resolve f-strings using already-collected vars
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and hasattr(node, 'lineno'):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.JoinedStr):
                        resolved = _resolve_ast_value(node.value, module_vars)
                        if resolved:
                            module_vars[target.id] = resolved

        for node in ast.walk(tree):
            te_list_node = None
            lineno = getattr(node, 'lineno', 0)

            # Pattern 1: extra_model_def["text_encoder_URLs"] = [...]
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        if (isinstance(target.slice, ast.Constant) and
                                target.slice.value == "text_encoder_URLs"):
                            te_list_node = node.value
                    elif isinstance(target, ast.Name) and "text_encoder" in target.id.lower():
                        te_list_node = node.value

            # Pattern 2: dict literal with "text_encoder_URLs" key
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (isinstance(key, ast.Constant) and key.value == "text_encoder_URLs"
                            and isinstance(value, ast.List)):
                        te_list_node = value
                        break

            if te_list_node is None:
                continue

            # Merge module-level vars with local vars (local takes precedence)
            local_vars = dict(module_vars)
            local_vars.update(_collect_local_vars(source, lineno))

            for url, te_folder in _extract_urls_from_list(te_list_node, local_vars):
                results.append({
                    'config_name': config_name,
                    'url': url,
                    'text_encoder_folder': te_folder,
                })

    return results


def _collect_local_vars(source: str, target_lineno: int) -> Dict[str, str]:
    """Collect string variable assignments closest to target_lineno (before it)."""
    # Collect all string assignments with their line numbers
    candidates: Dict[str, List[Tuple[int, str]]] = {}
    try:
        tree = ast.parse(source)
    except Exception:
        return {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not hasattr(node, 'lineno'):
            continue
        # Only look at assignments before the target within 30 lines
        if node.lineno > target_lineno or node.lineno < target_lineno - 30:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                name = target.id
                if name not in candidates:
                    candidates[name] = []
                candidates[name].append((node.lineno, node.value.value))

    # For each variable, pick the assignment closest to (but before) target_lineno
    local_vars = {}
    for name, assignments in candidates.items():
        closest = max(assignments, key=lambda x: x[0])
        local_vars[name] = closest[1]

    return local_vars


def _resolve_ast_value(node: ast.expr, local_vars: Dict[str, str]) -> Optional[str]:
    """Resolve an AST node to a string value."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in local_vars:
        return local_vars[node.id]
    # Handle f-strings like f"{_GEMMA_FOLDER}.safetensors"
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                resolved = _resolve_ast_value(value.value, local_vars)
                if resolved is None:
                    return None
                parts.append(resolved)
            else:
                return None
        return "".join(parts)
    return None


def update_supplemental_models(wan2gp_dir: Path, existing_urls: set = None) -> int:
    """Scan handlers and rebuild supplemental_models table.

    Clears and repopulates on every call so dropped models don't linger.

    Args:
        wan2gp_dir: Path to Wan2GP directory
        existing_urls: URLs already known from defaults/*.json (to avoid duplicates)

    Returns: number of models in table after scan
    """
    scanned = scan_handler_text_encoders(wan2gp_dir)

    conn = sqlite3.connect(QUEUE_DB_PATH)
    cursor = conn.cursor()

    # Clear and rebuild — keeps table in sync with current handler code
    cursor.execute("DELETE FROM supplemental_models")

    added = 0
    for item in scanned:
        url = item['url']
        # Skip if already in defaults JSON configs
        if existing_urls and url in existing_urls:
            continue

        try:
            cursor.execute('''
                INSERT OR IGNORE INTO supplemental_models
                (config_name, url, url_type, text_encoder_folder)
                VALUES (?, ?, 'TEXT_ENC', ?)
            ''', (item['config_name'], url, item['text_encoder_folder']))
            if cursor.rowcount > 0:
                added += 1
        except Exception:
            pass

    conn.commit()
    conn.close()
    return added
