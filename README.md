# WanGP Smart Model Downloader

A Python-based intelligent model downloader that integrates with your local model hub to avoid duplicate downloads.

## Features

- 🔍 **Real-time hub scanning** - Scans `/mnt/llm/hub/models` each run for latest files
- 🔗 **Smart symlinking** - Creates symlinks for existing files instead of downloading duplicates  
- 📊 **Live progress view** - Real-time status updates for all downloads
- 🚦 **Bandwidth limiting** - Configurable download speed limits
- 🔄 **Resume support** - Automatically resumes interrupted downloads
- 📈 **Progress tracking** - Visual progress bars and status indicators

## Installation

```bash
cd Wan2GP_model_download
./installd.sh
# Or manually:
# mkuv wandown-env 3.12.10
# source .envrc
# uv pip install -r requirements.txt
```

## Configuration

On first run, a `config.json` file is automatically created with default settings:

```json
{
  "hub_directory": "/mnt/llm/hub/models",
  "bandwidth_limit_kb": 90000,
  "wan2gp_directory": "../Wan2GP/ckpts",
  "_comment": "Edit this file to customize downloader settings. Set hub_directory to empty string to disable hub features."
}
```

**Configuration Options:**
- `hub_directory`: Path to your local model hub (set to `""` or invalid path to disable hub features)
- `bandwidth_limit_kb`: Default bandwidth limit in KB/s (optional)
- `wan2gp_directory`: Where to store downloaded models (defaults to `../Wan2GP/ckpts`)
- `_comment`: Helpful reminder text (ignored by the app)

**To disable hub features:** Set `"hub_directory": ""` in the config file.

## Usage

```bash
# Basic usage (uses config file settings)
python downloader.py

# Custom bandwidth limit (overrides config)
python downloader.py --limit 80000

# No bandwidth limit
python downloader.py --no-limit

# Custom hub directory (overrides config)
python downloader.py --hub-dir /path/to/your/hub

# Use different config file
python downloader.py --config-file my_config.json
```

## GUI Interface

The downloader uses a terminal-based interface (TUI) with the following behavior:

### Navigation & Selection
- **Arrow Keys**: Navigate through the model list
- **Click/Scroll**: Navigate to any row (no selection)
- **Enter**: Toggle selection checkbox for current row
- **Mouse Click**: Only moves cursor - does NOT select items

### Hotkeys
- **`d`** - Download all selected models
- **`a`** - Select all visible models  
- **`u`** - Unselect all models
- **`s`** - Toggle between "Show All" and "Missing Only" view
- **`r`** - Reset cache (clears HuggingFace API cache)
- **`p`** - Toggle color palette
- **`Escape`** - Quit application

### Display Columns
- **✓** - Selection checkbox (filled when selected)
- **Config** - Model configuration type
- **Model File** - Filename of the model
- **Size (GB)** - File size in gigabytes
- **Status** - Current status (Pending, Downloading, Complete, etc.)
- **Destination** - Target directory for download

### Selection Behavior
- Only **Enter key** toggles selection - clicking does not select
- Selected items show a filled checkbox (✓)
- Navigate with arrows, select with Enter, download with 'd'

## Status Indicators

- **Pending** - Waiting to process
- **Exists** - File already exists in destination  
- **Linked** - Created symlink from hub
- **Downloading** - Currently downloading
- **Complete** - Successfully downloaded
- **Failed** - Download failed

## How It Works

1. **Loads configuration** from `config.json` (or uses defaults)
2. **Scans your model hub** (if configured and valid) for existing files
3. **Builds a hash database** mapping file hashes to hub locations
4. **For each selected model:**
   - Checks if file already exists at destination
   - If hub enabled: Gets file hash from HuggingFace API and checks hub
   - Creates symlink if found in hub, otherwise downloads
5. **Shows live progress** in the terminal interface

**Key Benefits:**
- **Smart hub integration** - Reuses existing files via symlinks to save space and bandwidth
- **Terminal interface** - Clean, keyboard-driven selection and monitoring
- **Bandwidth limiting** - Respects configured download speeds
- **Resume support** - Continues interrupted downloads
- **Real-time updates** - Live status and progress tracking

This saves bandwidth and storage by reusing files you already have!