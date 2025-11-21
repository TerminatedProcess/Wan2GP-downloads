#!/usr/bin/env python3
"""
WanGP Smart Model Downloader - Textual UI Version
Stable terminal interface that won't get corrupted
"""

import os
import sys
import json
import requests
import httpx
import time
import threading
import sqlite3
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Optional

from huggingface_hub import hf_hub_download, HfApi
try:
    from huggingface_hub import set_client_factory
except ImportError:
    # Fallback for older versions
    try:
        from huggingface_hub import configure_http_backend
        set_client_factory = None
    except ImportError:
        set_client_factory = None
        configure_http_backend = None

import click
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, DataTable, Static, ProgressBar, Log
from textual.reactive import reactive
from textual import work, events
from textual.worker import get_current_worker

# Configure logging to reset log file on each app start
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('downloader.log', mode='w'),  # 'w' mode overwrites file each time
    ]
)

class BandwidthLimitedSession(requests.Session):
    """Requests session with bandwidth limiting (for legacy support)"""

    def __init__(self, max_bytes_per_second: Optional[int] = None):
        super().__init__()
        self.max_bytes_per_second = max_bytes_per_second

    def request(self, method, url, **kwargs):
        # Let HuggingFace handle streaming, don't interfere with it
        response = super().request(method, url, **kwargs)

        # Only apply throttling to actual file downloads (large responses)
        if (self.max_bytes_per_second and
            hasattr(response, 'headers') and
            response.headers.get('content-length') and
            int(response.headers.get('content-length', 0)) > 1024 * 1024):  # > 1MB

            # Apply throttling by adding delays during iteration
            original_iter_content = response.iter_content

            def throttled_iter_content(chunk_size=1024, decode_unicode=False):
                start_time = time.time()
                bytes_downloaded = 0

                for chunk in original_iter_content(chunk_size=chunk_size, decode_unicode=decode_unicode):
                    if chunk:
                        bytes_downloaded += len(chunk)
                        yield chunk

                        # Calculate if we need to sleep
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            expected_time = bytes_downloaded / self.max_bytes_per_second
                            if expected_time > elapsed:
                                time.sleep(expected_time - elapsed)

            response.iter_content = throttled_iter_content

        return response

class BandwidthLimitedTransport(httpx.HTTPTransport):
    """HTTPX transport with bandwidth limiting (for huggingface_hub v1.0+)"""

    def __init__(self, max_bytes_per_second: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.max_bytes_per_second = max_bytes_per_second
        self._start_time = None
        self._bytes_downloaded = 0

    def handle_request(self, request):
        # Reset throttling state for new request
        self._start_time = time.time()
        self._bytes_downloaded = 0

        # Get response from parent transport
        response = super().handle_request(request)

        # Only throttle large responses (>1MB)
        content_length = response.headers.get('content-length')
        if (self.max_bytes_per_second and content_length and
            int(content_length) > 1024 * 1024):

            # Wrap the response stream to add throttling
            original_stream = response.stream

            def throttled_stream():
                for chunk in original_stream:
                    if chunk:
                        self._bytes_downloaded += len(chunk)
                        yield chunk

                        # Apply throttling
                        elapsed = time.time() - self._start_time
                        if elapsed > 0:
                            expected_time = self._bytes_downloaded / self.max_bytes_per_second
                            if expected_time > elapsed:
                                time.sleep(expected_time - elapsed)

            # Replace stream with throttled version
            response.stream = throttled_stream()

        return response

class ModelDownloader:
    def __init__(self, hub_dir: str = None, bandwidth_limit: Optional[int] = None, cache_dir: str = None, config_file: str = "config.json"):
        # Load configuration from file
        self.config = self.load_config(config_file)
        
        # Use parameters if provided, otherwise fall back to config, then defaults
        self.hub_dir = Path(hub_dir) if hub_dir else Path(self.config.get("hub_directory", "/mnt/llm/hub/models"))
        self.cache_dir = Path(cache_dir) if cache_dir else Path(self.config.get("wan2gp_directory", "../Wan2GP/ckpts"))
        self.bandwidth_limit = bandwidth_limit if bandwidth_limit is not None else self.config.get("bandwidth_limit_kb")
        
        # Check if hub directory exists and is valid
        self.hub_enabled = self.hub_dir.exists() and self.hub_dir.is_dir()
        if not self.hub_enabled:
            print(f"Hub directory not found or invalid: {self.hub_dir} - Hub features disabled")
        
        self.hash_db = {}
        self.download_queue = []
        self.hf_api = HfApi()
        self.cache_db_path = "hfcache.db"
        self.init_cache_db()  # Initialize SQLite cache database

        # Configure bandwidth-limited HTTP backend if limit specified
        if self.bandwidth_limit:
            if set_client_factory is not None:
                # Use new API for huggingface_hub v1.0+
                def create_httpx_client():
                    transport = BandwidthLimitedTransport(
                        max_bytes_per_second=self.bandwidth_limit * 1024  # Convert KB/s to bytes/s
                    )
                    return httpx.Client(transport=transport)
                set_client_factory(create_httpx_client)
            elif configure_http_backend is not None:
                # Fallback for older versions
                def create_session():
                    return BandwidthLimitedSession(max_bytes_per_second=self.bandwidth_limit * 1024)
                configure_http_backend(backend_factory=create_session)
            else:
                print("Warning: Could not configure bandwidth limiting - huggingface_hub API not available")
    
    def load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file, create if missing"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default config file on first run
                default_config = self.create_default_config()
                print(f"Config file {config_file} not found, creating with defaults...")
                try:
                    with open(config_path, 'w') as f:
                        json.dump(default_config, f, indent=2)
                    print(f"Created {config_file} - you can edit it to customize settings")
                except Exception as write_error:
                    print(f"Warning: Could not create config file {config_file}: {write_error}")
                return default_config
        except Exception as e:
            print(f"Error loading config file {config_file}: {e} - using defaults")
            return self.create_default_config()
    
    def create_default_config(self) -> dict:
        """Create default configuration"""
        return {
            "hub_directory": "/mnt/llm/hub/models",
            "bandwidth_limit_kb": 90000,
            "wan2gp_directory": "../Wan2GP/ckpts",
            "_comment": "Edit this file to customize downloader settings. Set hub_directory to empty string to disable hub features."
        }
    
    def init_cache_db(self):
        """Initialize SQLite cache database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
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
            
            # Create index for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_repo_file 
                ON hf_file_cache(repo_id, filename)
            ''')
            
            conn.commit()
            conn.close()
            
            # Count existing entries
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM hf_file_cache WHERE success = 1")
            count = cursor.fetchone()[0]
            conn.close()
            
            if count > 0:
                print(f"Loaded HF file info cache with {count} entries")
            else:
                print("Initialized new HF cache database")
                
        except Exception as e:
            print(f"Warning: Could not initialize cache database: {e}")
    
    def get_cached_file_info(self, cache_key: str) -> dict:
        """Get cached file info from database - returns dict with 'found' and 'data' keys"""
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
                    # Cached failure - don't retry
                    return {'found': True, 'data': None}
            
            # Not in cache
            return {'found': False, 'data': None}
            
        except Exception as e:
            print(f"Warning: Error reading from cache database: {e}")
            return {'found': False, 'data': None}
    
    def cache_file_info(self, cache_key: str, repo_id: str, filename: str, file_info=None, raw_data: str = None):
        """Cache file info to database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            if file_info:
                # Successful API call
                file_size = file_info.size if hasattr(file_info, 'size') else None
                # Handle different LFS object structures in different huggingface_hub versions
                file_hash = None
                if hasattr(file_info, 'lfs') and file_info.lfs:
                    if hasattr(file_info.lfs, 'oid'):
                        file_hash = file_info.lfs.oid
                    elif hasattr(file_info.lfs, 'sha256'):
                        file_hash = file_info.lfs.sha256
                    else:
                        # Try to get any hash-like attribute
                        for attr in dir(file_info.lfs):
                            if not attr.startswith('_') and attr.lower() in ['oid', 'sha256', 'hash']:
                                file_hash = getattr(file_info.lfs, attr, None)
                                break
                success = 1
                logging.debug(f"Caching successful API result for {cache_key} (size: {file_size})")
            else:
                # Failed API call
                file_size = None
                file_hash = None
                success = 0
                logging.debug(f"Caching failed API result for {cache_key}")
                
            cursor.execute('''
                INSERT OR REPLACE INTO hf_file_cache 
                (cache_key, repo_id, filename, file_size, file_hash, raw_data, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (cache_key, repo_id, filename, file_size, file_hash, raw_data, success))
            
            conn.commit()
            conn.close()
            logging.debug(f"Successfully wrote cache record for {cache_key}")
            
        except Exception as e:
            print(f"Warning: Error writing to cache database: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_cache_db(self):
        """Clear all entries from cache database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM hf_file_cache")
            conn.commit()
            conn.close()
            print("Cache database cleared")
        except Exception as e:
            print(f"Warning: Error clearing cache database: {e}")
    
    def delete_cache_db(self):
        """Delete the entire cache database file"""
        try:
            cache_file = Path(self.cache_db_path)
            if cache_file.exists():
                cache_file.unlink()
                print("Cache database file deleted")
            # Reinitialize
            self.init_cache_db()
        except Exception as e:
            print(f"Warning: Error deleting cache database: {e}")
        
    def scan_hub(self) -> Dict[str, Dict]:
        """Scan the model hub and build hash database"""
        hash_db = {}
        
        if not self.hub_enabled:
            return hash_db
            
        # Scan all directories in hub (each directory name is a hash)
        for hash_dir in self.hub_dir.iterdir():
            if hash_dir.is_dir():
                hash_value = hash_dir.name
                
                # Find model files in this hash directory
                model_files = []
                for ext in ['*.safetensors', '*.gguf', '*.bin', '*.pth']:
                    model_files.extend(hash_dir.glob(ext))
                
                # Store info for each file
                for model_file in model_files:
                    filename = model_file.name
                    file_size = model_file.stat().st_size
                    
                    hash_db[hash_value] = {
                        'path': str(model_file),
                        'filename': filename,
                        'size': file_size
                    }
        
        self.hash_db = hash_db
        return hash_db
    
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
        """Get file info from HuggingFace API with SQLite caching"""
        repo_id, filename = self.parse_hf_url(url)
        if not repo_id or not filename:
            return None
        
        # Create cache key
        cache_key = f"{repo_id}/{filename}"
        
        # Check cache first
        cache_result = self.get_cached_file_info(cache_key)
        if cache_result['found']:
            if cache_result['data']:
                print(f"Cache hit for {cache_key}")
                return cache_result['data']
            else:
                print(f"Cache hit (failure) for {cache_key}")
                return None
        
        # Cache miss - need to fetch from API
        print(f"Fetching file info from HF API: {cache_key}")
            
        try:
            file_info_list = self.hf_api.get_paths_info(repo_id, [filename])
            if file_info_list and len(file_info_list) > 0:
                info = file_info_list[0]
                
                # Extract all available attributes safely
                raw_data_dict = {
                    'type': str(type(info)),
                    'available_attributes': dir(info)
                }
                
                # Add attributes that exist
                safe_attrs = ['size', 'path', 'blob_id', 'lfs']
                for attr in safe_attrs:
                    if hasattr(info, attr):
                        try:
                            value = getattr(info, attr)
                            if attr == 'lfs' and value:
                                # Handle LFS object safely
                                raw_data_dict['lfs'] = {
                                    'type': str(type(value)),
                                    'attributes': dir(value)
                                }
                                if hasattr(value, 'oid'):
                                    raw_data_dict['lfs']['oid'] = value.oid
                                if hasattr(value, 'size'):
                                    raw_data_dict['lfs']['size'] = value.size
                            else:
                                raw_data_dict[attr] = value
                        except Exception as e:
                            raw_data_dict[f'{attr}_error'] = str(e)
                
                raw_data = json.dumps(raw_data_dict, indent=2)
                
                # Extract hash safely - handle different LFS object structures
                file_hash = None
                if hasattr(info, 'lfs') and info.lfs:
                    if hasattr(info.lfs, 'oid'):
                        file_hash = info.lfs.oid
                    elif hasattr(info.lfs, 'sha256'):
                        file_hash = info.lfs.sha256
                
                # Cache the successful result
                self.cache_file_info(cache_key, repo_id, filename, info, raw_data)
                
                print(f"Successfully cached file info for {cache_key}: {info.size} bytes")
                
                return {
                    'size': info.size,
                    'hash': file_hash,
                    'raw_data': raw_data
                }
        except Exception as e:
            print(f"HF API call failed for {cache_key}: {type(e).__name__}: {str(e)}")
            # Cache the failure to avoid repeated API calls for same file
            self.cache_file_info(cache_key, repo_id, filename, file_info=None, raw_data=str(e))
            
        return None
    
    def get_hf_file_hash(self, url: str) -> Optional[str]:
        """Get file hash from HuggingFace API with caching"""
        file_info = self.get_hf_file_info(url)
        return file_info['hash'] if file_info else None
    
    def get_hf_file_size(self, url: str) -> Optional[int]:
        """Get file size from HuggingFace API with caching"""
        file_info = self.get_hf_file_info(url)
        return file_info['size'] if file_info else None
    
    def find_in_hub(self, file_hash: str) -> Optional[Dict]:
        """Find file in hub by hash"""
        if file_hash in self.hash_db:
            return self.hash_db[file_hash]
        return None
    
    def create_symlink(self, source_path: str, target_path: str) -> tuple[bool, str]:
        """Create symlink from source to target"""
        try:
            target = Path(target_path)
            source = Path(source_path)
            
            # Check if source actually exists
            if not source.exists():
                return False, f"Source file does not exist: {source_path}"
            
            target.parent.mkdir(parents=True, exist_ok=True)
            
            if target.exists() or target.is_symlink():
                target.unlink()
            
            target.symlink_to(source)
            
            # Verify symlink was created successfully
            if target.is_symlink() and target.exists():
                return True, "Symlink created successfully"
            else:
                return False, f"Symlink verification failed for {target_path}"
            
        except Exception as e:
            return False, f"Symlink creation failed: {str(e)}"
    
    def determine_output_path(self, url: str, url_type: str) -> str:
        """Determine the correct output path based on file type and URL"""
        filename = Path(urlparse(url).path).name
        
        if any(pattern in filename.lower() for pattern in ['lora', 'adapter']):
            if 'flux' in filename.lower():
                return f"../Wan2GP/loras_flux/{filename}"
            elif 'hunyuan' in filename.lower():
                return f"../Wan2GP/loras_hunyuan/{filename}"
            elif 'i2v' in filename.lower():
                return f"../Wan2GP/loras_i2v/{filename}"
            elif 'ltxv' in filename.lower() or 'ltx' in filename.lower():
                return f"../Wan2GP/loras_ltxv/{filename}"
            elif 'qwen' in filename.lower():
                return f"../Wan2GP/loras_qwen/{filename}"
            else:
                return f"../Wan2GP/loras/{filename}"
        else:
            return f"../Wan2GP/ckpts/{filename}"
    
    def resolve_config_urls(self, config_file: str) -> List[Tuple[str, str, str]]:
        """Resolve URLs from config file, handling references"""
        def resolve_urls(urls, defaults_dir='../Wan2GP/defaults'):
            if isinstance(urls, str):
                ref_file = Path(defaults_dir) / f"{urls}.json"
                if ref_file.exists():
                    try:
                        with open(ref_file, 'r') as f:
                            ref_config = json.load(f)
                        ref_urls = ref_config.get('model', {}).get('URLs', [])
                        return resolve_urls(ref_urls, defaults_dir)
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
        """Get multiple file sizes from cache in a single database query"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            # Create placeholders for the IN clause
            placeholders = ','.join('?' * len(cache_keys))
            
            cursor.execute(f'''
                SELECT cache_key, file_size
                FROM hf_file_cache 
                WHERE cache_key IN ({placeholders}) AND success = 1
            ''', cache_keys)
            
            results = cursor.fetchall()
            conn.close()
            
            # Return as dictionary
            return {cache_key: file_size for cache_key, file_size in results}
            
        except Exception as e:
            print(f"Warning: Error reading batch from cache database: {e}")
            return {}
    
    def build_download_queue(self, progress_callback=None) -> List[Dict]:
        """Build complete download queue from all config files"""
        queue = []
        defaults_dir = Path("../Wan2GP/defaults")
        
        if not defaults_dir.exists():
            return queue
        
        # First pass: collect all URLs and build basic queue
        url_to_cache_key = {}
        for config_file in sorted(defaults_dir.glob("*.json")):
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
                    'status': 'pending',
                    'progress': 0,
                    'remote_size': None  # Will be filled in batch
                })
        
        # Second pass: batch lookup ALL file sizes from cache (regardless of filter)
        if url_to_cache_key:
            cache_keys = list(url_to_cache_key.values())
            cached_sizes = self.get_batch_file_sizes(cache_keys)
            
            print(f"Loaded {len(cached_sizes)} file sizes from cache in batch")
            if progress_callback:
                logging.debug(f"Progress callback available: {progress_callback}")
            else:
                logging.debug("No progress callback")
            
            # Apply sizes to ALL queue items and track cache misses
            cache_misses = []
            for item in queue:
                url = item['url']
                if url in url_to_cache_key:
                    cache_key = url_to_cache_key[url]
                    if cache_key in cached_sizes:
                        item['remote_size'] = cached_sizes[cache_key]
                    else:
                        # Cache miss - need to fetch from API
                        cache_misses.append((item, cache_key, url))
            
            # Fetch missing sizes from HuggingFace API with progress indicator
            if cache_misses:
                total_misses = len(cache_misses)
                logging.debug(f"Found {total_misses} cache misses, building cache...")
                if progress_callback:
                    logging.debug(f"Calling progress_callback(0, {total_misses}, 'Building HuggingFace API Cache')")
                    progress_callback(0, total_misses, "Building HuggingFace API Cache")
                else:
                    print(f"Building HuggingFace cache: fetching {total_misses} file sizes...")
                
                # Group cache misses by repo to batch API calls efficiently
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
                        print(f"Fetching batch of {len(batch_items)} files from {repo_id}...")
                        filenames = [filename for _, _, filename, _ in batch_items]
                        file_info_list = self.hf_api.get_paths_info(repo_id, filenames)
                        
                        # Create lookup by filename
                        info_by_filename = {info.path: info for info in file_info_list if hasattr(info, 'path')}
                        
                        for item, cache_key, filename, url in batch_items:
                            processed += 1
                            if progress_callback:
                                progress_callback(processed, total_misses, f"Building HuggingFace API Cache")
                            else:
                                print(f"Building HuggingFace cache: {processed}/{total_misses} - {cache_key}")
                            
                            if filename in info_by_filename:
                                info = info_by_filename[filename]
                                if hasattr(info, 'size'):
                                    # Cache this successful result
                                    raw_data_dict = {'size': info.size, 'path': filename}
                                    if hasattr(info, 'lfs') and info.lfs:
                                        # Handle different LFS object structures
                                        lfs_data = {}
                                        if hasattr(info.lfs, 'oid'):
                                            lfs_data['oid'] = info.lfs.oid
                                        elif hasattr(info.lfs, 'sha256'):
                                            lfs_data['sha256'] = info.lfs.sha256
                                        raw_data_dict['lfs'] = lfs_data
                                    raw_data = json.dumps(raw_data_dict)
                                    
                                    self.cache_file_info(cache_key, repo_id, filename, info, raw_data)
                                    item['remote_size'] = info.size
                                    print(f"✓ Got size for {filename}: {info.size} bytes")
                                else:
                                    # Cache failure
                                    self.cache_file_info(cache_key, repo_id, filename, None, "No size attribute")
                                    print(f"✗ No size for {filename}")
                            else:
                                # File not found in batch response
                                self.cache_file_info(cache_key, repo_id, filename, None, "File not found in batch")
                                print(f"✗ File not found: {filename}")
                        
                    except Exception as e:
                        # Cache all failures in this batch
                        for item, cache_key, filename, url in batch_items:
                            processed += 1
                            if progress_callback:
                                progress_callback(processed, total_misses, f"Building HuggingFace API Cache")
                            else:
                                print(f"Building HuggingFace cache: {processed}/{total_misses} - {cache_key} (FAILED)")
                            self.cache_file_info(cache_key, repo_id, filename, None, str(e))
                            if not progress_callback:
                                print(f"✗ Batch failed for {filename}: {e}")
                
                if progress_callback:
                    progress_callback(total_misses, total_misses, "HuggingFace API Cache Complete")
                else:
                    print("HuggingFace cache building complete!")
        
        # Third pass: Check hub for missing files and auto-create symlinks
        if self.hub_enabled:
            print("Checking hub for existing files and creating symlinks...")
            symlinks_created = 0
            items_checked = 0
            for item in queue:
                output_path = Path(item['output_path'])
                
                # Skip if file already exists or is already symlinked
                if output_path.exists():
                    if output_path.is_symlink():
                        item['status'] = 'symlinked'
                    else:
                        item['status'] = 'exists'
                    continue
                
                # Only check hub for HTTP URLs
                if not item['url'].startswith('http'):
                    continue
                
                # Get file hash from cache if available
                repo_id, filename = self.parse_hf_url(item['url'])
                if repo_id and filename:
                    cache_key = f"{repo_id}/{filename}"
                    cached_info = self.get_cached_file_info(cache_key)
                    items_checked += 1
                    
                    if cached_info['found'] and cached_info['data'] and cached_info['data']['hash']:
                        file_hash = cached_info['data']['hash']
                        
                        # Look for file in hub
                        hub_file = self.find_in_hub(file_hash)
                        if hub_file and Path(hub_file['path']).exists():
                            # Create symlink
                            success, msg = self.create_symlink(hub_file['path'], str(output_path))
                            if success:
                                item['status'] = 'symlinked'
                                symlinks_created += 1
                                print(f"✓ Symlinked {item['filename']} from hub")
                            else:
                                print(f"✗ Failed to symlink {item['filename']}: {msg}")
                        else:
                            print(f"  Hash {file_hash[:8]}... not found in hub for {item['filename']}")
                    else:
                        print(f"  No cached hash for {item['filename']}")
            
            print(f"Hub check complete: {items_checked} items checked, {symlinks_created} symlinks created")
            if symlinks_created > 0:
                print(f"Created {symlinks_created} symlinks from hub files")
        
        return queue
    
    def download_hf_file(self, url: str, output_path: str, item: Dict) -> tuple[bool, str]:
        """Download file from HuggingFace using hf_hub_download"""
        try:
            repo_id, filename = self.parse_hf_url(url)
            if not repo_id or not filename:
                return False, f"Failed to parse URL: {url}"
            
            # Create output directory
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download to HuggingFace cache first
            cached_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir)
            )
            
            # Verify the cached file actually exists
            if not Path(cached_file).exists():
                return False, f"HuggingFace returned non-existent file: {cached_file}"
            
            # Create symlink from cache to destination
            success, msg = self.create_symlink(cached_file, output_path)
            if success:
                return True, "Success"
            else:
                return False, msg
            
        except Exception as e:
            return False, f"Download error: {str(e)}"

class DownloaderApp(App):
    """Textual app for the downloader"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    .title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin: 1 0;
        padding: 1;
    }
    
    .controls {
        dock: top;
        height: 3;
        background: $panel;
        padding: 0 1;
    }
    
    .status-bar {
        dock: bottom;
        height: 3;
        background: $panel;
        padding: 0 1;
    }
    
    DataTable {
        margin: 0 1;
        scrollbar-background: $panel;
        scrollbar-color: $accent;
    }
    
    DataTable .datatable--column-2 {
        width: 50%;
    }
    
    DataTable .datatable--column-3 {
        width: 10%;
        text-align: right;
    }
    
    DataTable > .datatable--header {
        background: $primary;
        text-style: bold;
    }
    
    DataTable > .datatable--cursor {
        background: $accent;
        color: $text;
    }
    
    .status-pending { color: $warning; }
    .status-exists { color: $success; }
    .status-symlinked { color: cyan; }
    .status-downloading { color: $primary; text-style: bold; }
    .status-completed { color: $success; text-style: bold; }
    .status-failed { color: $error; text-style: bold; }
    
    .status-messages {
        height: 3;
        margin: 0 1;
        padding: 1;
        background: $surface-darken-1;
        text-style: bold;
    }
    
    .main-status {
        background: $surface-darken-1;
        text-style: bold;
        color: $text;
    }
    
    .progress-info {
        color: $warning;
        text-align: right;
        min-width: 15;
    }
    
    .progress-bar {
        width: 30%;
        margin: 0 1;
        visibility: hidden;
    }
    
    Log {
        border: none;
        scrollbar-background: $panel;
        scrollbar-color: $accent;
    }
    """
    
    BINDINGS = [
        ("escape", "smart_escape", "Abort/Quit"),
        ("enter", "toggle_current_row", "Select"),
        ("d", "download_selected", "Download"),
        ("a", "select_all", "Select All"),
        ("u", "unselect_all", "Unselect All"),
        ("s", "toggle_show_all", "Show All"),
        ("r", "reset_cache", "Reset Cache"),
    ]
    
    def __init__(self, hub_dir: str = None, bandwidth_limit: Optional[int] = None, config_file: str = "config.json"):
        super().__init__()
        self.downloader = ModelDownloader(
            hub_dir=hub_dir,
            bandwidth_limit=bandwidth_limit,
            config_file=config_file
        )
        self.is_downloading = False
        self.config_checkboxes = {}  # Store checkboxes by config name
        self.selected_items = set()
        self.show_all_files = False  # Filter flag - False means hide existing files
    
    def format_file_size(self, size_bytes: Optional[int]) -> str:
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
        
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, name="WanGP Model Downloader")
        
        # Top status message area
        with Horizontal(classes="status-messages"):
            yield Static("Ready - Use hotkeys below for actions", id="main-status", classes="main-status")
            yield ProgressBar(id="progress-bar", classes="progress-bar")
            yield Static("", id="progress-info", classes="progress-info")
        
        # Main data table
        yield DataTable(id="download-table")
        
        # Bottom log bar  
        with Horizontal(classes="status-bar"):
            yield Log(id="log", auto_scroll=True)
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the interface"""
        # Show initializing status immediately
        logging.debug("Initializing... (on_mount)")
        main_status = self.query_one("#main-status", Static)
        main_status.update("Initializing...")
        logging.debug("Widget updated to 'Initializing...' (on_mount)")
        
        self.setup_table()
        self.scan_hub()
    
    def setup_table(self):
        """Setup the download table"""
        table = self.query_one("#download-table", DataTable)
        table.add_columns("✓", "Config", "Model File", "Size (GB)", "Status", "Destination")
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.show_cursor = True
        table.focus()
    
    @work(exclusive=True, thread=True)
    async def scan_hub(self):
        """Scan the hub in a worker thread"""
        
        # Check database existence first and update status
        db_exists = Path(self.downloader.cache_db_path).exists()
        self.call_from_thread(self.update_status_message, f"Database exists: {db_exists}")
        
        # Run scanning directly (already in worker thread)
        self.call_from_thread(self.update_status_message, "Scanning hub directory...")
        hash_db = self.downloader.scan_hub()
        
        # Build download queue with progress callback
        self.call_from_thread(self.update_status_message, "Building download queue...")
        def progress_update(current, total, message):
            # Update progress bar and status
            self.call_from_thread(self.update_cache_progress, current, total, message)
        
        queue = self.downloader.build_download_queue(progress_callback=progress_update)
        self.downloader.download_queue = queue
        
        # Hide progress and update table
        self.call_from_thread(self.hide_progress)
        self.call_from_thread(self.populate_table)
    
    def update_status_message(self, message):
        """Update the main status message"""
        logging.debug(f"Status update: {message}")
        main_status = self.query_one("#main-status", Static)
        main_status.update(message)
        logging.debug(f"Widget updated to '{message}'")
    
    def update_cache_progress(self, current, total, message):
        """Update the progress bar and status for cache building"""
        logging.debug(f"update_cache_progress called: {current}/{total} - {message}")
        main_status = self.query_one("#main-status", Static)
        progress_bar = self.query_one("#progress-bar", ProgressBar)
        progress_info = self.query_one("#progress-info", Static)
        
        # Show progress bar
        progress_bar.styles.visibility = "visible"
        
        # Update progress bar
        progress_bar.update(total=total, progress=current)
        
        # Update status text
        main_status.update(message)
        progress_info.update(f"{total}:{current}")
    
    def hide_progress(self):
        """Hide the progress bar and reset status"""
        progress_bar = self.query_one("#progress-bar", ProgressBar)
        progress_info = self.query_one("#progress-info", Static)
        main_status = self.query_one("#main-status", Static)
        
        # Hide progress bar
        progress_bar.styles.visibility = "hidden"
        
        # Clear progress info
        progress_info.update("")
        
        # Reset status to ready
        main_status.update("Ready - Use hotkeys below for actions")
    
    def get_filtered_items(self):
        """Get filtered items based on current filter settings"""
        items_to_show = []
        for item in self.downloader.download_queue:
            # Check if file exists
            file_exists = Path(item['output_path']).exists()
            
            # Update item status if it exists but wasn't marked as such
            if file_exists and item['status'] == 'pending':
                if Path(item['output_path']).is_symlink():
                    item['status'] = 'symlinked'
                else:
                    item['status'] = 'exists'
            
            # Consider files with these statuses as "existing"
            considered_existing = item['status'] in ['exists', 'symlinked', 'completed']
            
            # Filter logic: show all files OR show only non-existing files
            if self.show_all_files or not considered_existing:
                items_to_show.append(item)
        return items_to_show
    
    def populate_table(self):
        """Populate the download table with clean formatting"""
        table = self.query_one("#download-table", DataTable)
        table.clear()
        
        # Get filtered items
        items_to_show = self.get_filtered_items()
        
        for item in items_to_show:
            # Clean destination path
            destination = item['output_path'].replace("../Wan2GP/", "").split("/")[0]
            
            # Clean status display
            status = item['status']
            status_map = {
                'pending': 'Pending',
                'exists': 'Exists', 
                'symlinked': 'Linked',
                'downloading': 'Downloading',
                'completed': 'Complete',
                'failed': 'Failed'
            }
            status_display = status_map.get(status, status.title())
            
            # Selection indicator with red color
            selected = item['output_path'] in self.selected_items
            check_mark = "[red]✓[/red]" if selected else ""
            
            # Clean filename display - no truncation
            filename = item['filename']
            
            # Get file size - check local file first, then remote size
            file_size = None
            if Path(item['output_path']).exists():
                # Get local file size
                file_size = Path(item['output_path']).stat().st_size
            else:
                # Use remote file size from API
                file_size = item.get('remote_size')
            
            size_display = self.format_file_size(file_size)
            
            # Log if size is unknown for troubleshooting (to console, not UI)
            if size_display == "Unknown":
                logging.debug(f"Unknown size for {item['filename']}, remote_size: {item.get('remote_size')}")
            
            table.add_row(
                check_mark,
                item['config'],
                filename,
                size_display,
                status_display,
                destination
            )
        
        # Update progress info
        total_files = len(self.downloader.download_queue)
        filtered_count = len(items_to_show)
        selected_count = len(self.selected_items)
        progress_info = self.query_one("#progress-info", Static)
        
        if self.show_all_files:
            progress_info.update(f"{selected_count}/{total_files} selected")
        else:
            progress_info.update(f"{selected_count}/{filtered_count} selected ({total_files - filtered_count} hidden)")
        
        
        table.refresh()
        
        # Fix startup rendering issue - force complete screen refresh after initial population
        if not hasattr(self, '_initial_refresh_done'):
            self._initial_refresh_done = True
            # Force table to be properly rendered before user interaction
            table.focus()  # Ensure table has focus for navigation
            self.refresh()
            self.screen.refresh()
            # Small delay to ensure rendering is complete
            self.call_later(self._final_startup_refresh)
    
    def update_progress_info(self):
        """Update the progress info display with current selection count"""
        total_files = len(self.downloader.download_queue)
        filtered_count = len(self.get_filtered_items())
        selected_count = len(self.selected_items)
        progress_info = self.query_one("#progress-info", Static)
        
        if self.show_all_files:
            progress_info.update(f"{selected_count}/{total_files} selected")
        else:
            progress_info.update(f"{selected_count}/{filtered_count} selected ({total_files - filtered_count} hidden)")
    
    def _final_startup_refresh(self):
        """Final refresh after startup to ensure table is properly rendered"""
        table = self.query_one("#download-table", DataTable)
        table.refresh()
        self.refresh()
    
    def toggle_item_selection(self, item: Dict):
        """Toggle selection for a specific item and handle dependencies."""
        output_path = item['output_path']
        is_currently_selected = output_path in self.selected_items

        # Toggle the primary item
        if is_currently_selected:
            self.selected_items.remove(output_path)
        else:
            self.selected_items.add(output_path)
        
        # Handle high/low dependency
        filename = item['filename'].lower()
        counterpart_path = None
        
        # Normalize to handle cases like 'low-res' or 'high-res'
        if 'high' in filename:
            counterpart_filename = item['filename'].lower().replace('high', 'low')
        elif 'low' in filename:
            counterpart_filename = item['filename'].lower().replace('low', 'high')
        else:
            counterpart_filename = None

        if counterpart_filename:
            # Find the counterpart item in the full download queue
            for potential_mate in self.downloader.download_queue:
                if potential_mate['filename'].lower() == counterpart_filename:
                    counterpart_path = potential_mate['output_path']
                    break
        
        if counterpart_path:
            # Sync the counterpart's state with the primary item
            if is_currently_selected:
                # If we just deselected the primary, deselect the counterpart
                if counterpart_path in self.selected_items:
                    self.selected_items.remove(counterpart_path)
            else:
                # If we just selected the primary, select the counterpart
                if counterpart_path not in self.selected_items:
                    self.selected_items.add(counterpart_path)

        self.populate_table()
    
    
    def on_data_table_row_selected(self, message) -> None:
        """Handle table row clicks - cursor is already moved by DataTable"""
        table = self.query_one("#download-table", DataTable)
        current_row = table.cursor_row
        
        logging.debug(f"Table row selected: {current_row}, cursor position: {table.cursor_row}")
        
        # Mouse clicks should ONLY move the orange cursor, not toggle selection
        # The cursor movement is already handled by DataTable automatically
        # Only ENTER key should toggle selections via action_toggle_current_row()
        
        # Get the filtered items (what's actually displayed in the table)
        filtered_items = self.get_filtered_items()
        
        # Log the cursor movement for debugging
        if 0 <= current_row < len(filtered_items):
            selected_item = filtered_items[current_row]
            logging.debug(f"Mouse moved cursor to row {current_row}: {selected_item['filename']}")
        else:
            logging.debug(f"Invalid cursor position: {current_row}, max: {len(filtered_items)}")
    
    
    def action_select_all(self):
        """Select all visible items"""
        filtered_items = self.get_filtered_items()
        self.selected_items = {item['output_path'] for item in filtered_items}
        # Update status message
        main_status = self.query_one("#main-status", Static)
        main_status.update(f"Selected {len(self.selected_items)} items - Use hotkeys below for actions")
        
        self.populate_table()  # Refresh table to show updated selection
    
    def action_unselect_all(self):
        """Deselect all items"""
        self.selected_items.clear()
        # Update status message
        main_status = self.query_one("#main-status", Static)
        main_status.update("All items deselected - Use hotkeys below for actions")
        
        self.populate_table()  # Refresh table to show updated selection
    
    def action_toggle_current_row(self):
        """Toggle selection for the currently selected table row"""
        table = self.query_one("#download-table", DataTable)
        if table.cursor_row is not None:
            # Store cursor position
            cursor_position = table.cursor_row
            # Need to find the actual item from filtered list since table only shows filtered items
            filtered_items = self.get_filtered_items()
            if cursor_position < len(filtered_items):
                selected_item = filtered_items[cursor_position]
                self.toggle_item_selection(selected_item)
                
                # Store cursor position before rebuilding table
                cursor_position = table.cursor_row
                
                # Rebuild the table to update the checkbox
                self.populate_table()
                
                # Restore cursor position using move_cursor method
                if cursor_position is not None and cursor_position < table.row_count:
                    # Use the DataTable's move_cursor method to restore position
                    for _ in range(cursor_position):
                        table.action_cursor_down()
                
                # Log the action for debugging
                log = self.query_one("#log", Log)
                log.write_line(f"Toggled row {cursor_position}: {selected_item['config']}")
    
    def action_toggle_show_all(self):
        """Toggle between showing all files vs only missing files"""
        self.show_all_files = not self.show_all_files
        
        # Refresh table with new filter
        self.populate_table()
        
        # Update status message and log the change
        main_status = self.query_one("#main-status", Static)
        log = self.query_one("#log", Log)
        if self.show_all_files:
            main_status.update("Showing all files - Use hotkeys below for actions")
            log.write_line("Display mode: all files")
        else:
            main_status.update("Showing missing files only - Use hotkeys below for actions")
            log.write_line("Display mode: missing files only")
    
    def action_reset_cache(self):
        """Reset cache - delete database and rebuild app"""
        # Update status messages
        main_status = self.query_one("#main-status", Static)
        main_status.update("Resetting cache and rebuilding...")
        
        # Delete and reinitialize cache database
        self.downloader.delete_cache_db()
        
        log = self.query_one("#log", Log)
        log.write_line("Cache database reset - rebuilding application")
        
        # Clear current state
        self.selected_items.clear()
        
        # Rebuild the download queue (this will refetch all file sizes)
        self.scan_hub()
    
    def action_toggle_palette(self):
        """Toggle the command palette"""
        self.action_command_palette()
    
    def action_clear_cache(self):
        """Clear the HuggingFace API cache and rebuild"""
        # Delete and reinitialize cache database
        self.downloader.delete_cache_db()
        
        log = self.query_one("#log", Log)
        log.write_line("Cache database deleted and reinitialized")
        
        # Update status messages
        main_status = self.query_one("#main-status", Static)
        main_status.update("Rebuilding cache...")
        
        # Rebuild the download queue (this will refetch all file sizes)
        self.scan_hub()
        
        # Reset status messages
        main_status.update("Cache rebuilt - Use hotkeys below for actions")
        log = self.query_one("#log", Log)
        log.write_line("Cache cleared - refetching file sizes from HuggingFace")
    
    def action_smart_escape(self):
        """Smart escape: abort downloads if running, otherwise quit"""
        if self.is_downloading:
            self.action_abort_downloads()
        else:
            self.exit()
    
    def action_abort_downloads(self):
        """Immediately abort all downloads"""
        if self.is_downloading:
            self.is_downloading = False
            log = self.query_one("#log", Log)
            log.write_line("❌ DOWNLOAD ABORTED by user")
            
            # Mark any downloading items as failed
            for item in self.downloader.download_queue:
                if item['status'] == 'downloading':
                    item['status'] = 'pending'  # Reset to pending so user can retry
            
            self.populate_table()
    
    def action_download_selected(self):
        """Start downloads for selected configs only"""
        if not self.selected_items:
            log = self.query_one("#log", Log)
            log.write_line("❌ No models selected! Please select models to download.")
            return
            
        if not self.is_downloading:
            self.start_selected_downloads()
    
    def action_start(self):
        """Start downloads"""
        if not self.is_downloading:
            self.start_downloads()
    
    def action_cancel(self):
        """Cancel downloads"""
        if self.is_downloading:
            self.is_downloading = False
    
    
    
    @work(exclusive=True, thread=True)
    async def process_single_item(self, item):
        """Process a single download item"""
        if item['status'] in ['completed', 'exists', 'symlinked']:
            self.call_from_thread(
                self.query_one("#log", Log).write_line, 
                f"Item already processed: {item['filename']}"
            )
            return
            
        self.is_downloading = True
        filename = item['filename']
        url = item['url']
        output_path = item['output_path']
        
        # Update current item status
        item['status'] = 'downloading'
        self.call_from_thread(self.populate_table)
        
        
        self.call_from_thread(self.query_one("#log", Log).write_line, f"Processing: {filename}")
        self.call_from_thread(self.query_one("#status", Static).update, f"Processing: {filename[:30]}...")
        
        # Check if file already exists
        if Path(output_path).exists() and not Path(output_path).is_symlink():
            item['status'] = 'exists'
            self.call_from_thread(self.query_one("#log", Log).write_line, f"Already exists: {filename}")
        else:
            # Get file hash from HuggingFace and check hub if enabled
            if self.downloader.hub_enabled:
                file_hash = self.downloader.get_hf_file_hash(url)
                
                if file_hash:
                    # Look for file in hub
                    hub_file = self.downloader.find_in_hub(file_hash)
                    
                    if hub_file:
                        # Create symlink
                        success, msg = self.downloader.create_symlink(hub_file['path'], output_path)
                        if success:
                            item['status'] = 'symlinked'
                            self.call_from_thread(self.query_one("#log", Log).write_line, f"Symlinked from hub: {filename}")
                            # Update table and finish processing this item
                            self.call_from_thread(self.populate_table)
                            self.is_downloading = False
                            return
                        else:
                            item['status'] = 'failed'
                            self.call_from_thread(self.query_one("#log", Log).write_line, f"Failed to create symlink: {filename} - {msg}")
            
            # Download file using HuggingFace Hub
            self.call_from_thread(self.query_one("#log", Log).write_line, f"Downloading: {filename}")
            
            success, error_msg = self.downloader.download_hf_file(url, output_path, item)
            if success:
                item['status'] = 'completed'
                item['progress'] = 100
                self.call_from_thread(self.query_one("#log", Log).write_line, f"Completed: {filename}")
            else:
                item['status'] = 'failed'
                self.call_from_thread(self.query_one("#log", Log).write_line, f"Failed: {filename} - {error_msg}")
        
        self.call_from_thread(self.populate_table)
        self.call_from_thread(self.query_one("#status", Static).update, "Ready")
        
        self.is_downloading = False
    
    def action_quit(self):
        """Quit the application"""
        self.exit()
    
    async def on_key(self, event: events.Key) -> None:
        """Handle key presses"""
        if event.key == "enter":
            # Make sure ENTER toggles the current row, not the select all checkbox
            table = self.query_one("#download-table", DataTable)
            if table.has_focus:
                self.action_toggle_current_row()
                event.prevent_default()
                return
        
    
    @work(exclusive=True, thread=True)
    async def start_downloads(self):
        """Start the download process"""
        self.is_downloading = True
        
        for i, item in enumerate(self.downloader.download_queue):
            if not self.is_downloading:
                break
                
            # Update current item status
            item['status'] = 'downloading'
            self.call_from_thread(self.populate_table)
            
            filename = item['filename']
            url = item['url']
            output_path = item['output_path']
            
            self.call_from_thread(self.query_one("#log", Log).write_line, f"Processing: {filename}")
            
            # Check if file already exists
            if Path(output_path).exists() and not Path(output_path).is_symlink():
                item['status'] = 'exists'
                self.call_from_thread(self.query_one("#log", Log).write_line, f"Already exists: {filename}")
                continue
            
            # Get file hash from HuggingFace
            file_hash = self.downloader.get_hf_file_hash(url)
            
            if file_hash:
                # Look for file in hub
                hub_file = self.downloader.find_in_hub(file_hash)
                
                if hub_file:
                    # Create symlink
                    success = self.downloader.create_symlink(hub_file['path'], output_path)
                    if success:
                        item['status'] = 'symlinked'
                        self.call_from_thread(self.query_one("#log", Log).write_line, f"Symlinked from hub: {filename}")
                        continue
            
            # Download file using HuggingFace Hub  
            self.call_from_thread(self.query_one("#log", Log).write_line, f"Downloading: {filename}")
            
            success = self.downloader.download_hf_file(url, output_path, item)
            if success:
                item['status'] = 'completed'
                item['progress'] = 100
                self.call_from_thread(self.query_one("#log", Log).write_line, f"Completed: {filename}")
            else:
                item['status'] = 'failed'
                self.call_from_thread(self.query_one("#log", Log).write_line, f"Failed: {filename}")
            
            self.call_from_thread(self.populate_table)
        self.call_from_thread(self.query_one("#status", Static).update, "Ready")
        
        self.is_downloading = False
    
    
    @work(exclusive=True, thread=True)
    async def start_selected_downloads(self):
        """Start downloads for selected configs only"""
        self.is_downloading = True
        
        selected_items = [
            item for item in self.downloader.download_queue 
            if item['output_path'] in self.selected_items
        ]
        
        
        self.call_from_thread(
            self.query_one("#log", Log).write_line,
            f"Starting downloads for {len(self.selected_items)} selected files..."
        )
        
        for i, item in enumerate(selected_items):
            if not self.is_downloading:
                break
                
            # Update current item status
            item['status'] = 'downloading'
            self.call_from_thread(self.populate_table)
            
            filename = item['filename']
            url = item['url']
            output_path = item['output_path']
            
            self.call_from_thread(self.query_one("#log", Log).write_line, f"Processing: {filename}")
            
            # Check if file already exists
            if Path(output_path).exists() and not Path(output_path).is_symlink():
                item['status'] = 'exists'
                self.call_from_thread(self.query_one("#log", Log).write_line, f"Already exists: {filename}")
                continue
            
            # Check hub if enabled
            if self.downloader.hub_enabled:
                # Get file hash from HuggingFace
                file_hash = self.downloader.get_hf_file_hash(url)
                
                if file_hash:
                    # Look for file in hub
                    hub_file = self.downloader.find_in_hub(file_hash)
                    
                    if hub_file:
                        # Create symlink
                        success, msg = self.downloader.create_symlink(hub_file['path'], output_path)
                        if success:
                            item['status'] = 'symlinked'
                            self.call_from_thread(self.query_one("#log", Log).write_line, f"Symlinked from hub: {filename}")
                            continue
            
            # Download file using HuggingFace Hub  
            self.call_from_thread(self.query_one("#log", Log).write_line, f"Downloading: {filename}")
            
            success, error_msg = self.downloader.download_hf_file(url, output_path, item)
            if success:
                item['status'] = 'completed'
                item['progress'] = 100
                self.call_from_thread(self.query_one("#log", Log).write_line, f"Completed: {filename}")
            else:
                item['status'] = 'failed'
                self.call_from_thread(self.query_one("#log", Log).write_line, f"Failed: {filename} - {error_msg}")
            
            self.call_from_thread(self.populate_table)
        self.call_from_thread(self.query_one("#status", Static).update, "Ready")
        
        self.is_downloading = False
        completed_count = sum(1 for item in selected_items if item['status'] in ['exists', 'symlinked', 'completed'])
        self.call_from_thread(
            self.query_one("#log", Log).write_line,
            f"Selected downloads complete! ({completed_count}/{len(selected_items)} successful)"
        )

def download_single_file(downloader: ModelDownloader, config_name: str, filename: str):
    """Download a single file without TUI"""
    print(f"Looking for {filename} in config {config_name}...")
    
    # Find the item in the queue
    target_item = None
    for item in downloader.download_queue:
        if item['config'] == config_name and filename in item['filename']:
            target_item = item
            break
    
    if not target_item:
        print(f"Error: Could not find {filename} in {config_name}")
        return False
    
    print(f"Found: {target_item['filename']}")
    print(f"URL: {target_item['url']}")
    print(f"Output: {target_item['output_path']}")
    
    # Check if already exists
    if Path(target_item['output_path']).exists():
        print("File already exists!")
        return True
    
    print("Starting download...")
    print("(This may take several minutes for large files...)")
    
    start_time = time.time()
    success, error_msg = downloader.download_hf_file(
        target_item['url'], 
        target_item['output_path'], 
        target_item
    )
    elapsed_time = time.time() - start_time
    
    if success:
        print(f"Download completed successfully in {elapsed_time:.1f} seconds!")
        if Path(target_item['output_path']).exists():
            file_size = Path(target_item['output_path']).stat().st_size
            print(f"File size: {file_size / (1024*1024):.1f} MB")
            print(f"Average speed: {(file_size / 1024) / elapsed_time:.1f} KB/s")
        return True
    else:
        print(f"Download failed after {elapsed_time:.1f} seconds: {error_msg}")
        return False

@click.command()
@click.option('--limit', '-l', type=int, help='Bandwidth limit in KB/s')
@click.option('--no-limit', is_flag=True, help='Download without bandwidth limits')
@click.option('--hub-dir', help='Path to model hub directory (overrides config file)')
@click.option('--config-file', default='config.json', help='Path to configuration file')
@click.option('--config', help='Download specific config (e.g. fun_inp)')
@click.option('--file', help='Download specific file pattern (e.g. Fun_InP)')
@click.option('--cli', is_flag=True, help='Use CLI mode instead of TUI')
def main(limit, no_limit, hub_dir, config_file, config, file, cli):
    """WanGP Smart Model Downloader with Textual UI or CLI mode"""
    
    bandwidth_limit = None if no_limit else limit
    
    if not Path("../Wan2GP/wgp.py").exists():
        print("Error: Please run this script from the Wan2GP_model_download directory next to WanGP")
        sys.exit(1)
    
    # CLI mode for direct downloads
    if cli or (config and file):
        print("WanGP Model Downloader - CLI Mode")
        
        downloader = ModelDownloader(
            hub_dir=hub_dir, 
            bandwidth_limit=bandwidth_limit,
            config_file=config_file
        )
        
        print(f"Bandwidth limit: {downloader.bandwidth_limit or 'unlimited'} KB/s")
        print(f"Hub directory: {downloader.hub_dir} ({'enabled' if downloader.hub_enabled else 'disabled'})")
        print(f"Wan2GP directory: {downloader.cache_dir}")
        
        print("Scanning hub...")
        downloader.scan_hub()
        
        print("Building download queue...")
        downloader.download_queue = downloader.build_download_queue()
        
        print(f"Found {len(downloader.download_queue)} files total")
        
        if config and file:
            success = download_single_file(downloader, config, file)
            sys.exit(0 if success else 1)
        else:
            print("Use --config and --file to specify what to download")
            sys.exit(1)
    
    # TUI mode
    app = DownloaderApp(
        hub_dir=hub_dir,
        bandwidth_limit=bandwidth_limit, 
        config_file=config_file
    )
    
    try:
        app.run()
    except (KeyboardInterrupt, SystemExit):
        # Clean up terminal mouse tracking on exit
        print('\033[?1000l\033[?1002l\033[?1003l\033[?1006l', end='', flush=True)
    finally:
        # Ensure mouse tracking is disabled on any exit
        print('\033[?1000l\033[?1002l\033[?1003l\033[?1006l', end='', flush=True)

if __name__ == "__main__":
    main()