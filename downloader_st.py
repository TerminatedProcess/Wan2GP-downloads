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
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

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

    def find_in_invokeai(self, url: str) -> Optional[str]:
        """Find model in InvokeAI database by source URL or filename"""
        if not self.invokeai_enabled:
            return None

        try:
            conn = sqlite3.connect(str(self.invokeai_db))
            cursor = conn.cursor()

            cursor.execute("SELECT path FROM models WHERE source = ?", (url,))
            result = cursor.fetchone()

            if not result:
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

        # Third pass: check InvokeAI
        if self.invokeai_enabled:
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

                invokeai_path = self.find_in_invokeai(item['url'])
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


def main():
    st.set_page_config(
        page_title="WanGP Model Downloader",
        page_icon="⬇️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'downloader' not in st.session_state:
        with st.spinner("Initializing downloader and loading models..."):
            st.session_state.downloader = ModelDownloader(config_file="config.yaml")

            # Auto-load queue on first startup (like Textual version)
            downloader = st.session_state.downloader
            st.session_state.download_queue = downloader.build_download_queue()

            st.session_state.selected_items = set()
            st.session_state.downloading = False

    downloader = st.session_state.downloader

    # Sidebar - Compact design
    with st.sidebar:
        st.markdown("### WanGP Model Downloader")

        if st.button("🔄 Refresh Queue", width="stretch", type="primary"):
            with st.spinner("Building download queue..."):
                progress_bar = st.progress(0, text="Building HuggingFace cache...")

                def progress_callback(current, total):
                    progress_bar.progress(current / total, text=f"Fetching metadata... {current}/{total}")

                st.session_state.download_queue = downloader.build_download_queue(progress_callback=progress_callback)
                progress_bar.empty()

            st.success(f"✓ Loaded {len(st.session_state.download_queue)} models", icon="✅")

        st.divider()

        # Bandwidth limit control
        st.markdown("**⚡ Bandwidth Limit**")
        current_limit_kb = downloader.bandwidth_limit if downloader.bandwidth_limit else 90000
        current_limit_mb = current_limit_kb / 1024  # Convert KB/s to MB/s

        bandwidth_mb = st.slider(
            "Download Speed (MB/s)",
            min_value=1,
            max_value=500,
            value=int(current_limit_mb),
            step=5,
            help="Limit download speed to prevent network saturation",
            label_visibility="collapsed"
        )

        st.caption(f"Current: {bandwidth_mb} MB/s (~{bandwidth_mb * 8:.0f} Mbps)")

        # Update bandwidth limit if changed
        new_limit_kb = bandwidth_mb * 1024
        if new_limit_kb != current_limit_kb:
            downloader.bandwidth_limit = new_limit_kb
            # Reconfigure HTTP backend
            if set_client_factory is not None:
                def create_httpx_client():
                    transport = BandwidthLimitedTransport(max_bytes_per_second=new_limit_kb * 1024)
                    return httpx.Client(transport=transport)
                set_client_factory(create_httpx_client)

        st.divider()

        # Filter
        filter_text = st.text_input("🔍 Filter models", placeholder="Type to search...", key="filter_input")

        show_all = st.checkbox("Show all files (including existing)", value=False, key="show_all")

        st.divider()

        if st.button("🗑️ Clear Cache", width="stretch"):
            with st.spinner("Clearing cache and rebuilding..."):
                downloader.clear_cache_db()

                # Rebuild queue immediately with progress
                progress_bar = st.progress(0, text="Fetching fresh metadata...")

                def progress_callback(current, total):
                    progress_bar.progress(current / total, text=f"Fetching metadata... {current}/{total}")

                st.session_state.download_queue = downloader.build_download_queue(progress_callback=progress_callback)
                progress_bar.empty()

            st.success("✓ Cache cleared and rebuilt!")
            st.rerun()

        # Stats at bottom
        if st.session_state.download_queue:
            total = len(st.session_state.download_queue)
            selected = len(st.session_state.selected_items)
            view_mode = "All files" if show_all else "Not Downloaded"
            st.caption(f"📊 Total: {total} models | View: {view_mode} | Selected: {selected}")

    # Main area - No huge header, just content
    if not st.session_state.download_queue:
        st.info("No models found in download queue. Check your config.yaml settings.")
        return

    # Filter queue
    filtered_queue = []
    for item in st.session_state.download_queue:
        # Update status if file exists
        file_exists = Path(item['output_path']).exists()
        if file_exists and item['status'] == '---':
            if Path(item['output_path']).is_symlink():
                item['status'] = 'Linked'
            else:
                item['status'] = 'Exists'

        # Filter by text
        if filter_text and filter_text.lower() not in item['filename'].lower():
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

        file_size = None
        if Path(item['output_path']).exists():
            file_size = Path(item['output_path']).stat().st_size
        else:
            file_size = item.get('remote_size')

        df_data.append({
            'Config': item['config'],
            'Model File': item['filename'],
            'Size (GB)': format_file_size(file_size),
            'Source': get_item_source(downloader, item),
            'Status': item['status'],
            'Dest': destination,  # Shortened column name
            '_output_path': item['output_path'],  # Hidden column for selection
        })

    df = pd.DataFrame(df_data)

    # Handle selection BEFORE rendering buttons (so counts are current)
    if 'model_table' in st.session_state:
        if 'selection' in st.session_state.model_table and 'rows' in st.session_state.model_table.selection:
            selected_rows = st.session_state.model_table.selection['rows']
            st.session_state.selected_items = set(df.iloc[selected_rows]['_output_path'].tolist())

    # Compact action bar directly above table
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 2])

    with col1:
        if st.button("✅ Select All", width="stretch", key="select_all"):
            st.session_state.selected_items = set(df['_output_path'].tolist())
            st.rerun()

    with col2:
        if st.button("❌ Clear", width="stretch", key="clear_all"):
            st.session_state.selected_items = set()
            st.rerun()

    with col3:
        selected_count = len(st.session_state.selected_items)
        if st.button(f"⬇️ Download Selected ({selected_count})", width="stretch", disabled=selected_count == 0, type="primary"):
            st.session_state.downloading = True
            st.rerun()

    with col4:
        # Empty spacer
        st.write("")

    with col5:
        st.markdown(f"**{selected_count}** of **{len(filtered_queue)}** selected")
        view_mode = "Showing All" if show_all else "Showing Not Downloaded"
        st.caption(view_mode)

    # Display table with selection - fill vertical space
    st.dataframe(
        df.drop(columns=['_output_path']),
        use_container_width=True,
        height=735,  # Tall table to fill screen
        selection_mode="multi-row",
        on_select="rerun",
        key="model_table",
        hide_index=True
    )

    # Download process
    if st.session_state.downloading:
        st.divider()
        st.subheader("Downloading...")

        selected_items = [
            item for item in filtered_queue
            if item['output_path'] in st.session_state.selected_items
        ]

        overall_progress = st.progress(0, text="Starting...")

        for i, item in enumerate(selected_items):
            filename = item['filename']
            url = item['url']
            output_path = item['output_path']

            # Use st.status for live updates
            with st.status(f"📥 Downloading: {filename[:50]}...", expanded=True) as status:
                # Check if already exists
                if Path(output_path).exists() and not Path(output_path).is_symlink():
                    item['status'] = 'Exists'
                    st.write("✓ File already exists")
                    status.update(label=f"✓ {filename[:50]}", state="complete")
                    continue

                # Check InvokeAI
                if downloader.invokeai_enabled:
                    invokeai_path = downloader.find_in_invokeai(url)
                    if invokeai_path:
                        success, msg = downloader.create_symlink(invokeai_path, output_path)
                        if success:
                            item['status'] = 'Linked'
                            st.write("✓ Symlinked from InvokeAI")
                            status.update(label=f"✓ {filename[:50]}", state="complete")
                            continue

                # Download with progress tracking
                st.write(f"Size: {format_file_size(item.get('remote_size'))} GB")
                file_progress = st.progress(0, text="Starting download...")

                # Store progress in session state for updates
                if 'download_progress' not in st.session_state:
                    st.session_state.download_progress = {}

                def progress_callback(downloaded, total, percent, speed_mb):
                    # Store in session state
                    st.session_state.download_progress = {
                        'downloaded': downloaded,
                        'total': total,
                        'percent': percent,
                        'speed_mb': speed_mb
                    }

                success, error_msg = downloader.download_hf_file(url, output_path, item, progress_callback=progress_callback)

                # Show final progress
                if 'download_progress' in st.session_state and st.session_state.download_progress:
                    prog = st.session_state.download_progress
                    downloaded_mb = prog['downloaded'] / (1024 * 1024)
                    total_mb = prog['total'] / (1024 * 1024)
                    file_progress.progress(1.0, text=f"✓ {downloaded_mb:.0f}/{total_mb:.0f} MB complete")

                if success:
                    item['status'] = 'Complete'
                    st.write("✓ Download complete")
                    status.update(label=f"✓ {filename[:50]}", state="complete")
                else:
                    item['status'] = 'Failed'
                    st.write(f"✗ Error: {error_msg}")
                    status.update(label=f"✗ {filename[:50]}", state="error")

            # Update overall progress
            overall_progress.progress((i + 1) / len(selected_items), text=f"Progress: {i + 1}/{len(selected_items)}")

        st.session_state.downloading = False
        st.success("Downloads complete!")

        if st.button("Done"):
            st.rerun()


if __name__ == "__main__":
    main()
