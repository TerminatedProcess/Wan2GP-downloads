#!/usr/bin/env python3
"""
HuggingFace Download Queue Processor
Standalone process that monitors and processes download queue independently.
Can be launched from command line or by Streamlit.
"""

import os
import sys
import json
import yaml
import time
import sqlite3
import signal
import argparse
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from urllib.parse import urlparse

import requests
import httpx
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


# Queue database path (shared with Streamlit app)
QUEUE_DB_PATH = "hfcache.db"


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


class QueueProcessor:
    """Processes download queue from database"""

    def __init__(self, config_file: str = "config.yaml", poll_interval: int = 5):
        self.config = self.load_config(config_file)
        self.poll_interval = poll_interval
        self.running = True
        self.current_job_id = None

        # Paths from config
        self.wan2gp_dir = Path(self.config.get("wan2gp_directory", "../Wan2GP-mryan"))
        self.cache_dir = self.wan2gp_dir / "ckpts"
        self.bandwidth_limit = self.config.get("bandwidth_limit_kb", 90000)

        # InvokeAI integration
        self.invokeai_db = Path(self.config.get("invokeai_db", "")) if self.config.get("invokeai_db") else None
        self.invokeai_models_dir = Path(self.config.get("invokeai_models_dir", "")) if self.config.get("invokeai_models_dir") else None
        self.invokeai_enabled = (
            self.invokeai_db is not None and
            self.invokeai_db.exists() and
            self.invokeai_models_dir is not None and
            self.invokeai_models_dir.exists()
        )

        # Hash index for SHA256 lookups
        self.hash_index = None
        if self.invokeai_enabled:
            try:
                from hash_index import HashIndex
                self.hash_index = HashIndex(
                    str(self.invokeai_db),
                    str(self.invokeai_models_dir),
                    self.config.get("parallel_hash_workers", 8)
                )
            except ImportError:
                pass

        # HuggingFace API
        self.hf_api = HfApi()

        # Configure bandwidth-limited HTTP backend
        self._configure_bandwidth_limit()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Received shutdown signal, finishing current job...")
        self.running = False

    def _configure_bandwidth_limit(self):
        """Configure bandwidth limiting for HuggingFace downloads"""
        # Note: Bandwidth limiting via custom transport can interfere with
        # HuggingFace's retry mechanism. For now, we skip custom transport
        # and rely on system-level throttling if needed (e.g., trickle, tc).
        # The bandwidth_limit config is still read but not applied at HTTP level.
        pass

    def load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading config: {e}")
        return {"wan2gp_directory": "../Wan2GP-mryan", "bandwidth_limit_kb": 90000}

    def init_queue_table(self):
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

    def reset_stale_downloads(self):
        """Reset any 'downloading' jobs back to 'pending' on startup"""
        conn = sqlite3.connect(QUEUE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE download_queue
            SET status = 'pending', progress = 0, speed_mbps = 0
            WHERE status = 'downloading'
        ''')

        count = cursor.rowcount
        conn.commit()
        conn.close()

        if count > 0:
            print(f"Reset {count} interrupted download(s) to pending")

    def get_next_job(self) -> Optional[Dict]:
        """Get the next pending job from the queue"""
        conn = sqlite3.connect(QUEUE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, url, output_path, filename, config_name, remote_size, hub_source_path
            FROM download_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT 1
        ''')

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'id': row[0],
                'url': row[1],
                'output_path': row[2],
                'filename': row[3],
                'config_name': row[4],
                'remote_size': row[5],
                'hub_source_path': row[6]
            }
        return None

    def update_job_status(self, job_id: int, status: str, progress: int = 0,
                         speed_mbps: float = 0, error_message: str = None):
        """Update job status in database"""
        conn = sqlite3.connect(QUEUE_DB_PATH)
        cursor = conn.cursor()

        if status == 'downloading':
            cursor.execute('''
                UPDATE download_queue
                SET status = ?, progress = ?, speed_mbps = ?, started_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, progress, speed_mbps, job_id))
        elif status in ('complete', 'failed', 'linked'):
            cursor.execute('''
                UPDATE download_queue
                SET status = ?, progress = ?, speed_mbps = ?, error_message = ?,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, progress, speed_mbps, error_message, job_id))
        else:
            cursor.execute('''
                UPDATE download_queue
                SET status = ?, progress = ?, speed_mbps = ?, error_message = ?
                WHERE id = ?
            ''', (status, progress, speed_mbps, error_message, job_id))

        conn.commit()
        conn.close()

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
        if sha256_hash and self.hash_index:
            try:
                result = self.hash_index.lookup_by_sha256(sha256_hash)
                if result:
                    file_path = Path(result['file_path'])
                    if file_path.exists():
                        return str(file_path)
            except Exception:
                pass

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
            print(f"  Error querying InvokeAI database: {e}")
            return None

    def parse_hf_url(self, url: str) -> tuple:
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

    def create_symlink(self, source_path: str, target_path: str) -> tuple:
        """Create symlink from source to target"""
        try:
            target = Path(target_path).resolve()
            source = Path(source_path).resolve()

            if not source.exists():
                return False, f"Source file does not exist: {source_path}"

            target.parent.mkdir(parents=True, exist_ok=True)

            # Delete existing file/symlink
            if target.exists() or target.is_symlink():
                print(f"    Deleting existing: {target} (symlink={target.is_symlink()})")
                target.unlink()

            target.symlink_to(source)

            # Verify the symlink was created correctly
            if target.is_symlink() and target.exists():
                print(f"    Created symlink: {target} -> {source}")
                return True, "Symlink created successfully"
            else:
                # Debug info
                print(f"    VERIFY FAILED: is_symlink={target.is_symlink()}, exists={target.exists()}")
                return False, f"Symlink verification failed for {target_path}"

        except Exception as e:
            return False, f"Symlink creation failed: {str(e)}"

    def download_file(self, job: Dict) -> tuple:
        """Download a file from HuggingFace with progress tracking"""
        url = job['url']
        output_path = job['output_path']
        job_id = job['id']
        expected_size = job.get('remote_size') or 0

        # Check if file already exists
        if Path(output_path).exists():
            return True, "File already exists"

        repo_id, filename = self.parse_hf_url(url)
        if not repo_id or not filename:
            return False, f"Failed to parse URL: {url}"

        print(f"  Repo: {repo_id}")
        print(f"  File: {filename}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        download_result = {'cached_file': None, 'error': None}
        download_done = threading.Event()

        def do_download():
            try:
                print("  Starting hf_hub_download...")
                download_result['cached_file'] = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(self.cache_dir)
                )
                print(f"  Download returned: {download_result['cached_file']}")
            except Exception as e:
                download_result['error'] = str(e)
                print(f"  Download error: {e}")
            finally:
                download_done.set()

        download_thread = threading.Thread(target=do_download)
        download_thread.start()

        # Poll progress and update database
        start_time = time.time()
        cache_path = Path(self.cache_dir)
        last_size = 0
        last_update = 0

        while not download_done.is_set():
            # Only show detailed progress if we know the expected size
            if expected_size > 0:
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
                            percent = int((current_size / expected_size) * 100)
                            percent = min(percent, 99)

                            # Update database every second
                            if time.time() - last_update >= 1:
                                self.update_job_status(job_id, 'downloading', percent, speed_mb)
                                last_update = time.time()

                            # Print progress bar
                            downloaded_mb = current_size / (1024 * 1024)
                            total_mb = expected_size / (1024 * 1024)
                            bar_width = 40
                            filled = int(bar_width * percent / 100)
                            bar = '█' * filled + '░' * (bar_width - filled)
                            print(f"\r  [{bar}] {percent}% | {downloaded_mb:.1f}/{total_mb:.1f} MB | {speed_mb:.1f} MB/s", end='', flush=True)

                    except (FileNotFoundError, OSError):
                        pass
            else:
                # No size info - just show elapsed time
                elapsed = time.time() - start_time
                if time.time() - last_update >= 5:
                    print(f"  Downloading... ({elapsed:.0f}s elapsed)")
                    last_update = time.time()

            download_done.wait(timeout=0.5)

        if expected_size > 0:
            print()  # New line after progress bar

        download_thread.join()

        if download_result['error']:
            return False, f"Download error: {download_result['error']}"

        cached_file = download_result['cached_file']

        if not cached_file:
            return False, "Download returned no file path"

        if not Path(cached_file).exists():
            return False, f"HuggingFace returned non-existent file: {cached_file}"

        success, msg = self.create_symlink(cached_file, output_path)
        return success, msg if not success else "Success"

    def process_job(self, job: Dict):
        """Process a single download job.

        Logic:
        1. Check if model is available in InvokeAI hub (via hub_source_path or lookup)
        2. If available in hub AND destination file/symlink exists → delete it
        3. Create symlink from hub
        4. If not in hub → download from HuggingFace
        """
        job_id = job['id']
        filename = job['filename']
        self.current_job_id = job_id

        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"\n[{timestamp}] Processing: {filename}")
        print(f"  URL: {job['url'][:80]}...")
        print(f"  Destination: {job['output_path']}")

        if job.get('remote_size'):
            size_mb = job['remote_size'] / (1024 * 1024)
            print(f"  Size: {size_mb:.1f} MB")

        output_path = Path(job['output_path'])

        # Check for hub path - either pre-stored or lookup
        hub_path = job.get('hub_source_path')

        if hub_path:
            print(f"  Using pre-stored hub path: {Path(hub_path).name}")
            if not Path(hub_path).exists():
                print(f"  WARNING: Hub path no longer exists, will search...")
                hub_path = None

        if not hub_path and self.invokeai_enabled:
            print("  Checking InvokeAI hub...")
            hub_path = self.find_in_invokeai(job['url'])

        if hub_path:
            # Model available in hub - delete existing and create symlink
            print(f"  Found in hub: {Path(hub_path).name}")
            if output_path.exists() or output_path.is_symlink():
                print(f"  Removing existing: {output_path.name} (symlink={output_path.is_symlink()})")
                output_path.unlink()

            success, message = self.create_symlink(hub_path, job['output_path'])
            if success:
                self.update_job_status(job_id, 'linked', 100, 0)
                print(f"  ✓ Linked from hub (no download needed)")
                self.current_job_id = None
                return
            else:
                print(f"  Symlink failed: {message}, falling back to download...")

        # Not in hub or symlink failed - proceed with download
        self.update_job_status(job_id, 'downloading', 0, 0)

        success, message = self.download_file(job)

        if success:
            self.update_job_status(job_id, 'complete', 100, 0)
            print(f"  ✓ Complete: {message}")
        else:
            self.update_job_status(job_id, 'failed', 0, 0, message)
            print(f"  ✗ Failed: {message}")

        self.current_job_id = None

    def get_queue_stats(self) -> Dict:
        """Get queue statistics"""
        conn = sqlite3.connect(QUEUE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT status, COUNT(*)
            FROM download_queue
            GROUP BY status
        ''')

        stats = dict(cursor.fetchall())
        conn.close()

        return {
            'pending': stats.get('pending', 0),
            'downloading': stats.get('downloading', 0),
            'complete': stats.get('complete', 0),
            'failed': stats.get('failed', 0),
            'linked': stats.get('linked', 0)
        }

    def run(self):
        """Main loop - poll queue and process jobs"""
        self.init_queue_table()

        print("=" * 60)
        print("HuggingFace Download Queue Processor")
        print("=" * 60)
        print(f"Config: {self.config.get('wan2gp_directory', '../Wan2GP-mryan')}")
        print(f"Bandwidth limit: {self.bandwidth_limit} KB/s ({self.bandwidth_limit/1024:.0f} MB/s)")
        print(f"Poll interval: {self.poll_interval} seconds")
        if self.invokeai_enabled:
            print(f"InvokeAI Hub: ✓ Enabled (models linked from hub skip download)")
        else:
            print(f"InvokeAI Hub: ✗ Disabled")
        print("-" * 60)

        # Reset any interrupted downloads from previous runs
        self.reset_stale_downloads()

        print("Waiting for jobs... (Ctrl+C to stop)")
        print()

        while self.running:
            job = self.get_next_job()

            if job:
                self.process_job(job)

                # Show queue stats after each job
                stats = self.get_queue_stats()
                linked_str = f", {stats['linked']} linked" if stats['linked'] > 0 else ""
                print(f"\n  Queue: {stats['pending']} pending, {stats['complete']} complete{linked_str}, {stats['failed']} failed")
            else:
                # No jobs, wait and poll again
                for _ in range(self.poll_interval):
                    if not self.running:
                        break
                    time.sleep(1)

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Queue processor stopped.")


def clear_queue():
    """Clear all entries from the queue"""
    conn = sqlite3.connect(QUEUE_DB_PATH)
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='download_queue'")
    if not cursor.fetchone():
        print("Queue is already empty (table not initialized)")
        conn.close()
        return

    cursor.execute("DELETE FROM download_queue")
    conn.commit()
    count = cursor.rowcount
    conn.close()
    print(f"Cleared {count} entries from download queue")


def show_queue():
    """Show current queue status"""
    conn = sqlite3.connect(QUEUE_DB_PATH)
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='download_queue'")
    if not cursor.fetchone():
        print("Queue is empty (table not initialized yet)")
        conn.close()
        return

    cursor.execute('''
        SELECT id, filename, status, progress, speed_mbps, created_at
        FROM download_queue
        ORDER BY created_at DESC
        LIMIT 20
    ''')

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("Queue is empty")
        return

    print("\n" + "=" * 80)
    print(f"{'ID':<5} {'Filename':<40} {'Status':<12} {'Progress':<10} {'Speed':<10}")
    print("-" * 80)

    for row in rows:
        job_id, filename, status, progress, speed, created = row
        filename_short = filename[:38] + '..' if len(filename) > 40 else filename
        progress_str = f"{progress}%" if progress else "-"
        speed_str = f"{speed:.1f} MB/s" if speed else "-"
        print(f"{job_id:<5} {filename_short:<40} {status:<12} {progress_str:<10} {speed_str:<10}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='HuggingFace Download Queue Processor')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--poll', type=int, default=5, help='Poll interval in seconds')
    parser.add_argument('--clear', action='store_true', help='Clear the download queue')
    parser.add_argument('--status', action='store_true', help='Show queue status')

    args = parser.parse_args()

    if args.clear:
        clear_queue()
        return

    if args.status:
        show_queue()
        return

    processor = QueueProcessor(config_file=args.config, poll_interval=args.poll)
    processor.run()


if __name__ == "__main__":
    main()
