# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WanGP Smart Model Downloader - A Python TUI application for downloading AI models from HuggingFace with intelligent hub integration. It avoids duplicate downloads by creating symlinks to existing files in a local model hub.

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
# TUI mode (primary usage)
python downloader.py

# With bandwidth limit (KB/s)
python downloader.py --limit 90000

# CLI mode for specific file
python downloader.py --cli --config fun_inp --file Fun_InP
```

### Testing
```bash
# Run GUI tests using Textual's test framework
python test_gui_complete.py
```

## Architecture

### Core Components

**`downloader.py`** - Single-file application containing:
- `ModelDownloader` class: Core download logic, HuggingFace API integration, hub scanning, symlink creation
- `DownloaderApp` class: Textual TUI application with DataTable-based interface
- Bandwidth limiting via `BandwidthLimitedTransport` (httpx) and `BandwidthLimitedSession` (requests)

### Data Flow

1. Loads config from `config.json` (auto-created with defaults on first run)
2. Scans hub directory (`/mnt/llm/hub/models`) to build hash→file mapping
3. Parses model URLs from `../Wan2GP-main/defaults/*.json` config files
4. Caches HuggingFace API responses in `hfcache.db` (SQLite)
5. For each selected model: check existence → check hub by hash → download if needed

### Key Patterns

- **Symlink-first**: Creates symlinks from hub rather than downloading duplicates
- **SQLite caching**: `hfcache.db` stores HF API file metadata to avoid repeated API calls
- **Textual workers**: Downloads run in `@work(thread=True)` decorated async methods
- **High/Low model pairing**: Selecting a "high" model auto-selects its "low" counterpart

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

### TUI Features
- **Filter bar**: Type to filter models by filename (real-time filtering)
- **Hotkeys**:
  - `Enter`: Toggle selection on current row
  - `d`: Download selected models
  - `a`/`u`: Select/unselect all
  - `s`: Toggle show all vs missing only
  - `r`: Reset HuggingFace cache
  - `Escape`: Abort downloads or quit

## Dependencies

- **textual**: TUI framework
- **huggingface_hub**: HF API and downloads
- **httpx/requests**: HTTP with bandwidth limiting
- **click**: CLI argument parsing
- **pyyaml**: YAML config parsing
