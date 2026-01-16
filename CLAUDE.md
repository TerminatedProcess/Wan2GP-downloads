# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WanGP Smart Model Downloader - Downloads AI models from HuggingFace for Wan2GP video generation. Uses SHA256 hash matching to avoid duplicate downloads by creating symlinks to existing files in a local model hub (InvokeAI).

## Commands

```bash
# Setup (after source .salias)
mkenv                          # Create .venv with Python 3.12.10
install                        # Install dependencies via uv

# Running
run                            # Start Streamlit web UI (primary interface)
queue                          # Start queue processor (separate terminal)
qs                             # Check queue status
hfk                            # Kill queue processor
stop                           # Stop Streamlit

# Direct commands
uv run streamlit run downloader_st.py
uv run python hfqueue.py
uv run python hfqueue.py --status
uv run python hfqueue.py --clear

# Hash index (CLI)
uv run python hash_index.py --status    # Show hash index stats
uv run python hash_index.py --rebuild   # Rebuild from scratch
```

## Architecture

```
User → Streamlit UI → SQLite queue → hfqueue.py → HuggingFace/InvokeAI → symlink
```

**Two-process design**: Streamlit handles UI/selection, `hfqueue.py` handles downloads independently. Communication via SQLite `download_queue` table.

### Core Files

| File | Purpose |
|------|---------|
| `downloader_st.py` | Streamlit web UI, `ModelDownloader` class, queue management |
| `hfqueue.py` | Standalone download processor with progress bars |
| `hash_index.py` | SHA256 index for InvokeAI models (`HashIndex` class) |
| `downloader.py` | Legacy Textual TUI (not queue-based) |

### Model Resolution Priority

1. **SHA256 hash match** - HuggingFace LFS OID → `hash_sha256.db` lookup
2. **URL exact match** - Source URL in InvokeAI's `models.source` column
3. **Filename match** - Last resort pattern matching
4. **Download** - If nothing found, download from HuggingFace

### Database Files

| File | Tables | Purpose |
|------|--------|---------|
| `hfcache.db` | `hf_file_cache`, `download_queue` | HF metadata cache + job queue |
| `hash_sha256.db` | `hash_index` | SHA256→file path mapping for InvokeAI |

### Configuration

`config.yaml` keys:
- `wan2gp_directory`: Path to Wan2GP (reads `defaults/*.json` for model URLs)
- `bandwidth_limit_kb`: Download speed limit (KB/s)
- `invokeai_db`: Path to InvokeAI's `invokeai.db`
- `invokeai_models_dir`: Path to InvokeAI's models directory
- `parallel_hash_workers`: Threads for SHA256 computation (default: 8)

### Key Classes

**`ModelDownloader`** (`downloader_st.py`):
- `build_download_queue()`: Scans `defaults/*.json`, resolves URLs, checks InvokeAI
- `find_in_invokeai()`: Multi-strategy model lookup
- `create_symlink()`: Safe symlink creation with verification

**`QueueProcessor`** (`hfqueue.py`):
- `get_next_job()`: FIFO from `download_queue` table
- `process_job()`: Check hub first, download if needed
- `download_file()`: Threaded download with progress polling

**`HashIndex`** (`hash_index.py`):
- `sync_from_invokeai()`: Import models from InvokeAI DB
- `compute_pending_hashes()`: Parallel SHA256 computation
- `lookup_by_sha256()`: Fast hash-to-path lookup

### HIGH/LOW Model Pairing

Models with both high-precision (bf16/fp16) and quantized (quanto) variants are grouped in the UI. The `group_high_low_models()` function pairs configs with exactly 2 files into single selectable rows.
