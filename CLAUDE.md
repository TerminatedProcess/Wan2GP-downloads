# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WanGP Smart Model Downloader - A Python web application for downloading AI models from HuggingFace with intelligent hub integration. It avoids duplicate downloads by creating symlinks to existing files in a local model hub (InvokeAI).

## Commands

### Setup
```bash
# Create virtual environment and install dependencies
source .salias
mkenv
install
```

### Running
```bash
# Streamlit web UI (primary usage)
run
# or: uv run streamlit run downloader_st.py

# Queue processor (run in separate terminal)
queue
# or: uv run python hfqueue.py

# Queue status check
qs
# or: uv run python hfqueue.py --status

# Clear queue
uv run python hfqueue.py --clear
```

## Architecture

### Core Components

**`downloader_st.py`** - Streamlit web application:
- `ModelDownloader` class: Core download logic, HuggingFace API integration, symlink creation
- Streamlit UI with tabs for Models and Queue
- Queue management functions for adding/viewing/clearing jobs

**`hfqueue.py`** - Standalone queue processor:
- Runs independently from Streamlit
- Monitors SQLite queue table and processes downloads
- Console output with progress bars
- Can be launched from command line

### Data Flow

1. User selects models in Streamlit → adds to queue table
2. `hfqueue.py` polls queue table → downloads pending jobs
3. Downloads go to HuggingFace cache → symlinks created to output paths
4. User clicks Refresh in Queue tab to see progress

### Key Patterns

- **Queue-based**: Downloads run in separate process, don't block web UI
- **Symlink-first**: Creates symlinks from InvokeAI rather than downloading duplicates
- **SQLite caching**: `hfcache.db` stores HF API metadata and download queue
- **Bandwidth limiting**: Configurable download speed limits

### Configuration

`config.yaml`:
```yaml
# Path to Wan2GP installation (where defaults/ and ckpts/ are located)
wan2gp_directory: ../Wan2GP-mryan

# Bandwidth limit in KB/s (90000 = ~90 MB/s)
bandwidth_limit_kb: 90000

# InvokeAI integration - create symlinks to existing models instead of re-downloading
invokeai_db: /mnt/llm/hub/invokeai_data/databases/invokeai.db
invokeai_models_dir: /mnt/llm/hub/invokeai_data/models
```

### InvokeAI Integration

The downloader queries InvokeAI's SQLite database to find existing models. If a model exists in InvokeAI, a symlink is created instead of downloading.

- First tries exact `source` URL match in InvokeAI's `models` table
- Falls back to filename matching (for locally imported models)
- Creates symlinks from `invokeai_models_dir/{uuid}/model.safetensors` to Wan2GP's `ckpts/`

### Streamlit UI Features

**Models Tab:**
- Filter bar for searching models by filename
- Select All / Clear buttons
- "Add to Queue" to queue selected models
- Toggle to show all files vs missing only

**Queue Tab:**
- Status metrics (pending, downloading, complete, failed)
- Refresh button to update view
- Clear Complete/Failed/All buttons
- Queue items table with progress and speed

### Database Schema

`hfcache.db` contains two tables:
- `hf_file_cache`: Cached HuggingFace API metadata
- `download_queue`: Job queue with status, progress, timestamps

## Dependencies

- **streamlit**: Web UI framework
- **pandas**: Data display in tables
- **huggingface_hub**: HF API and downloads
- **httpx/requests**: HTTP with bandwidth limiting
- **pyyaml**: YAML config parsing
