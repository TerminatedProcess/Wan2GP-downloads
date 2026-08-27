# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WanGP Smart Model Downloader - Downloads AI models from HuggingFace for Wan2GP video generation. Uses SHA256 hash matching to avoid duplicate downloads by creating symlinks to existing files in a local model hub (HubRoot).

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
```

## Service

The UI runs as a systemd user service, registered in PortHub and grouped under
**Media Stack** on the Service Dashboard (http://localhost:8090).

| Layer | Value |
|-------|-------|
| Unit | `~/.config/systemd/user/wangp-downloader.service` (Manual Start — not enabled at boot) |
| URL | http://localhost:8505 |
| Lease | `8505 wangp-downloader` in `~/.config/porthub/leases.sh` |
| Group | `media-stack` in `~/.config/porthub_service_dash/groups.json` |

`svcrun` / `svcstop` control the service; `run` still launches a foreground
instance (same port, so it will refuse to start while the service is up).

The queue processor (`hfqueue.py`) is deliberately **not** a service — the UI
starts and stops it, and a systemd unit would fight the UI's `pkill`.

## Architecture

```
User → Streamlit UI → SQLite queue → hfqueue.py → HuggingFace/HubRoot → symlink
```

**Two-process design**: Streamlit handles UI/selection, `hfqueue.py` handles downloads independently. Communication via SQLite `download_queue` table in `hfcache.db`.

The Streamlit UI adds jobs to the queue; the queue processor polls for `pending` jobs every 5 seconds. Progress updates are written back to the DB and the UI polls every 3 seconds (`@st.fragment(run_every="3s")`). Model status checks run every 10 seconds.

### Queue Job Statuses

`pending` → `downloading` → `complete` | `linked` | `failed`

- **linked**: Model found in HubRoot, symlinked without downloading
- **complete**: Downloaded from HuggingFace and symlinked from HF cache
- Queue processor resets stale `downloading` jobs to `pending` on startup

### Core Files

| File | Purpose |
|------|---------|
| `shared.py` | Shared utilities: bandwidth limiting, URL parsing, symlink creation, HubRoot lookup, config loading, queue table init |
| `downloader_st.py` | Streamlit web UI, `ModelDownloader` class, queue management functions |
| `hfqueue.py` | Standalone download processor with progress bars (runs as separate process) |
| `downloader.py` | Legacy Textual TUI (not queue-based, pre-dates Streamlit version) |

`shared.py` contains all code shared between `downloader_st.py` and `hfqueue.py`: `QUEUE_DB_PATH`, `BandwidthLimited*` classes, `parse_hf_url()`, `create_symlink()`, `find_in_hub()`, `load_config()`, and `init_queue_table()`.

### Model Resolution Priority

1. **SHA256 hash match** - HuggingFace LFS OID → HubRoot `hash_sha256` column lookup
2. **Filename match** - Fallback pattern matching against HubRoot `filename` column
3. **Download** - If nothing found, download from HuggingFace

### HubRoot Integration

HubRoot stores models at `{hub_models_dir}/{blake3_hash}/{filename}`. The database (`hubrootv3.db`) has a `models` table with `hash_sha256`, `hash_blake3`, `filename`, and `deleted` columns. SHA256 hashes are pre-populated for all models, eliminating the need for a separate hash index.

### Storage Layout

`hfqueue.py` hardcodes its HF cache to `{wan2gp_directory}/ckpts`, so downloads
land in Wan2GP's own model dir — **not** the default `~/.cache/huggingface`.

That `ckpts` dir now lives on `/mnt/llm` and is symlinked back into place:

```
/home/dev/work/services/Wan2GP-mryan/ckpts
    -> /mnt/llm/unsloth/huggingface/wan2gp-ckpts
```

This matches the existing `~/.cache/huggingface -> /mnt/llm/unsloth/huggingface`
pattern. It exists because `/home` is a 100%-full btrfs volume with ~19
snapshots; ~20 GB models written there both exhausted it and got pinned by
snapshot retention. A 19.6 GiB download failed with an opaque
`Internal Writer Error: Background writer channel closed` from the Rust hf_xet
backend, which is what ENOSPC looks like from that layer.

HubRoot (`/mnt/llm/hub/hubmodels`) is a *source* for symlinks only — a HubRoot
miss always falls through to a real download into `ckpts`.

### Database Files

| File | Tables | Purpose |
|------|--------|---------|
| `hfcache.db` | `hf_file_cache`, `download_queue` | HF metadata cache + job queue |

Schema migrations handled inline (PRAGMA table_info checks).

### Configuration

`config.yaml` keys:
- `wan2gp_directory`: Path to Wan2GP (reads `defaults/*.json` for model URLs)
- `bandwidth_limit_kb`: Download speed limit (KB/s)
- `hub_db`: Path to HubRoot's `hubrootv3.db`
- `hub_models_dir`: Path to HubRoot's models directory

### Key Classes

**`ModelDownloader`** (`downloader_st.py`):
- `build_download_queue()`: Three-pass process: (1) collect URLs from `defaults/*.json`, (2) batch-fetch file sizes from HF API with caching, (3) check HubRoot and auto-create symlinks
- `_find_in_hub()`: Delegates to `shared.find_in_hub()` (SHA256 → filename lookup)
- `resolve_config_urls()`: Handles recursive JSON references (URLs can be strings pointing to other config files)

**`QueueProcessor`** (`hfqueue.py`):
- `get_next_job()`: FIFO from `download_queue` table (oldest pending first)
- `process_job()`: Check `hub_source_path` first, then HubRoot lookup, then download
- `download_file()`: Threaded HF download with `.incomplete` file progress polling

### HIGH/LOW Model Pairing

Models with both high-precision (bf16/fp16) and quantized (quanto) variants are grouped in the UI. The `group_high_low_models()` function pairs configs with exactly 2 files into single selectable rows. Configs with 3+ files show as individual entries.

### Streamlit UI Structure

- **Models tab**: Builds queue from `defaults/*.json`, groups HIGH/LOW pairs, supports multi-row selection via `st.dataframe`. Filter and "Show All" toggle in sidebar.
- **Queue tab**: Shows download queue with auto-refresh. "Queue Hub" button batch-adds HubRoot-available models. Processor start/stop controls.
- Session state stores: `downloader`, `download_queue`, `selected_items`
