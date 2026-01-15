#!/usr/bin/env python3
"""
SHA256 Hash Index for InvokeAI Models
Builds and maintains a SHA256 index of InvokeAI model files for fast lookup.
"""

import sqlite3
import hashlib
import asyncio
import aiofiles
from pathlib import Path
from typing import Optional, Dict, List, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


HASH_DB_PATH = "hash_sha256.db"
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for hashing


class HashIndex:
    def __init__(self, invokeai_db: str, invokeai_models_dir: str, parallel_workers: int = 4):
        self.invokeai_db = Path(invokeai_db)
        self.invokeai_models_dir = Path(invokeai_models_dir)
        self.parallel_workers = parallel_workers
        self.db_path = HASH_DB_PATH

        # Callbacks for progress reporting
        self.on_progress: Optional[Callable[[int, int, str], None]] = None

    def init_db(self):
        """Initialize the hash index database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hash_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invokeai_id TEXT UNIQUE NOT NULL,
                blake3_hash TEXT NOT NULL,
                sha256_hash TEXT,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                computed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sha256 ON hash_index(sha256_hash)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_blake3 ON hash_index(blake3_hash)
        ''')

        conn.commit()
        conn.close()

    def delete_db(self):
        """Delete the hash index database"""
        db_file = Path(self.db_path)
        if db_file.exists():
            db_file.unlink()

    def get_stats(self) -> Dict:
        """Get index statistics"""
        if not Path(self.db_path).exists():
            return {'total': 0, 'computed': 0, 'pending': 0, 'exists': False}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM hash_index")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM hash_index WHERE sha256_hash IS NOT NULL")
        computed = cursor.fetchone()[0]

        conn.close()

        return {
            'total': total,
            'computed': computed,
            'pending': total - computed,
            'exists': True
        }

    def is_ready(self) -> bool:
        """Check if all hashes have been computed"""
        stats = self.get_stats()
        return stats['exists'] and stats['pending'] == 0

    def sync_from_invokeai(self) -> int:
        """
        Sync models from InvokeAI database.
        Adds new models, removes deleted ones.
        Returns number of new entries added.
        """
        if not self.invokeai_db.exists():
            raise FileNotFoundError(f"InvokeAI database not found: {self.invokeai_db}")

        self.init_db()

        # Get all models from InvokeAI
        invokeai_conn = sqlite3.connect(str(self.invokeai_db))
        invokeai_cursor = invokeai_conn.cursor()

        invokeai_cursor.execute('''
            SELECT id, hash, path FROM models
        ''')

        invokeai_models = invokeai_cursor.fetchall()
        invokeai_conn.close()

        # Get existing entries in our index
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT invokeai_id FROM hash_index")
        existing_ids = set(row[0] for row in cursor.fetchall())

        # Add new models
        added = 0
        for invokeai_id, blake3_hash, rel_path in invokeai_models:
            if invokeai_id in existing_ids:
                continue

            full_path = self.invokeai_models_dir / rel_path

            # Get file size if file exists
            file_size = None
            if full_path.exists():
                file_size = full_path.stat().st_size

            cursor.execute('''
                INSERT INTO hash_index (invokeai_id, blake3_hash, file_path, file_size)
                VALUES (?, ?, ?, ?)
            ''', (invokeai_id, blake3_hash, str(full_path), file_size))
            added += 1

        # Remove entries for deleted models
        invokeai_ids = set(m[0] for m in invokeai_models)
        deleted_ids = existing_ids - invokeai_ids

        if deleted_ids:
            placeholders = ','.join('?' * len(deleted_ids))
            cursor.execute(f"DELETE FROM hash_index WHERE invokeai_id IN ({placeholders})",
                          list(deleted_ids))

        conn.commit()
        conn.close()

        return added

    def _compute_sha256(self, file_path: str) -> Optional[str]:
        """Compute SHA256 hash of a file"""
        path = Path(file_path)
        if not path.exists():
            return None

        # Skip directories - some InvokeAI models are folders, not single files
        if path.is_dir():
            return None

        sha256 = hashlib.sha256()

        try:
            with open(path, 'rb') as f:
                while chunk := f.read(CHUNK_SIZE):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except (IOError, OSError) as e:
            # Handle any file access errors gracefully
            return None

    async def _compute_sha256_async(self, file_path: str) -> Optional[str]:
        """Compute SHA256 hash of a file asynchronously"""
        path = Path(file_path)
        if not path.exists():
            return None

        # Skip directories - some InvokeAI models are folders, not single files
        if path.is_dir():
            return None

        sha256 = hashlib.sha256()

        try:
            async with aiofiles.open(path, 'rb') as f:
                while chunk := await f.read(CHUNK_SIZE):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except (IOError, OSError) as e:
            return None

    def _update_hash(self, entry_id: int, sha256_hash: str):
        """Update SHA256 hash in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE hash_index
            SET sha256_hash = ?, computed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (sha256_hash, entry_id))

        conn.commit()
        conn.close()

    def compute_pending_hashes(self, progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """
        Compute SHA256 hashes for all pending entries using thread pool.
        This is the synchronous version for simpler integration.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, file_path, file_size FROM hash_index
            WHERE sha256_hash IS NULL
            ORDER BY file_size ASC
        ''')

        pending = cursor.fetchall()
        conn.close()

        if not pending:
            return

        total = len(pending)
        completed = 0

        def process_entry(entry):
            nonlocal completed
            entry_id, file_path, file_size = entry
            filename = Path(file_path).name

            sha256 = self._compute_sha256(file_path)

            if sha256:
                self._update_hash(entry_id, sha256)

            completed += 1

            if progress_callback:
                progress_callback(completed, total, filename)

            return sha256 is not None

        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            results = list(executor.map(process_entry, pending))

        return sum(results)

    async def compute_pending_hashes_async(self,
                                           progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """
        Compute SHA256 hashes for all pending entries asynchronously.
        Uses semaphore to limit concurrent operations.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Order by file_size to process smaller files first (quicker feedback)
        cursor.execute('''
            SELECT id, file_path, file_size FROM hash_index
            WHERE sha256_hash IS NULL
            ORDER BY file_size ASC
        ''')

        pending = cursor.fetchall()
        conn.close()

        if not pending:
            return 0

        total = len(pending)
        completed = 0
        semaphore = asyncio.Semaphore(self.parallel_workers)

        async def process_entry(entry):
            nonlocal completed
            entry_id, file_path, file_size = entry
            filename = Path(file_path).name

            async with semaphore:
                # Run hash computation in thread pool to not block event loop
                loop = asyncio.get_event_loop()
                sha256 = await loop.run_in_executor(None, self._compute_sha256, file_path)

                if sha256:
                    # Update DB in thread pool too
                    await loop.run_in_executor(None, self._update_hash, entry_id, sha256)

                completed += 1

                if progress_callback:
                    progress_callback(completed, total, filename)

                return sha256 is not None

        tasks = [process_entry(entry) for entry in pending]
        results = await asyncio.gather(*tasks)

        return sum(results)

    def lookup_by_sha256(self, sha256_hash: str) -> Optional[Dict]:
        """
        Look up a model by its SHA256 hash.
        Returns dict with invokeai_id, blake3_hash, file_path if found.
        """
        if not Path(self.db_path).exists():
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT invokeai_id, blake3_hash, file_path, file_size
            FROM hash_index
            WHERE sha256_hash = ?
        ''', (sha256_hash,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'invokeai_id': row[0],
                'blake3_hash': row[1],
                'file_path': row[2],
                'file_size': row[3]
            }
        return None

    def lookup_by_blake3(self, blake3_hash: str) -> Optional[Dict]:
        """Look up a model by its BLAKE3 hash"""
        if not Path(self.db_path).exists():
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT invokeai_id, sha256_hash, file_path, file_size
            FROM hash_index
            WHERE blake3_hash = ?
        ''', (blake3_hash,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'invokeai_id': row[0],
                'sha256_hash': row[1],
                'file_path': row[2],
                'file_size': row[3]
            }
        return None


def rebuild_index(invokeai_db: str, invokeai_models_dir: str, parallel_workers: int = 4,
                  progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Dict:
    """
    Rebuild the hash index from scratch.
    Returns stats dict.
    """
    index = HashIndex(invokeai_db, invokeai_models_dir, parallel_workers)

    # Delete existing and recreate
    index.delete_db()

    # Sync from InvokeAI
    added = index.sync_from_invokeai()

    # Compute all hashes
    index.compute_pending_hashes(progress_callback)

    return index.get_stats()


if __name__ == "__main__":
    # CLI for testing
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='SHA256 Hash Index Builder')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild index from scratch')
    parser.add_argument('--status', action='store_true', help='Show index status')
    parser.add_argument('--workers', type=int, help='Override parallel workers')

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    invokeai_db = config.get('invokeai_db', '')
    invokeai_models_dir = config.get('invokeai_models_dir', '')
    parallel_workers = args.workers or config.get('parallel_hash_workers', 4)

    if not invokeai_db or not invokeai_models_dir:
        print("Error: invokeai_db and invokeai_models_dir must be set in config.yaml")
        exit(1)

    index = HashIndex(invokeai_db, invokeai_models_dir, parallel_workers)

    if args.status:
        stats = index.get_stats()
        print(f"Hash Index Status:")
        print(f"  Total entries: {stats['total']}")
        print(f"  Computed: {stats['computed']}")
        print(f"  Pending: {stats['pending']}")
        print(f"  Ready: {index.is_ready()}")
        exit(0)

    if args.rebuild:
        print(f"Rebuilding hash index with {parallel_workers} workers...")
        index.delete_db()

    # Sync from InvokeAI
    print("Syncing from InvokeAI database...")
    added = index.sync_from_invokeai()
    print(f"  Added {added} new entries")

    stats = index.get_stats()
    if stats['pending'] > 0:
        print(f"Computing SHA256 hashes for {stats['pending']} files...")

        def progress(current, total, filename):
            pct = int(current / total * 100)
            print(f"\r  [{pct:3d}%] {current}/{total} - {filename[:50]:<50}", end='', flush=True)

        index.compute_pending_hashes(progress_callback=progress)
        print()

    stats = index.get_stats()
    print(f"\nComplete! {stats['computed']}/{stats['total']} hashes computed")
