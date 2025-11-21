# Textual TUI Application Testing Guide

## The Problem
When developing Text User Interface (TUI) applications with Textual, Claude Code cannot normally test interactive functionality because:
- Background processes can't receive keyboard input
- Terminal escape sequences are displayed as raw text
- No way to simulate keypresses or user interactions
- Cannot verify that keys like ENTER, arrows, etc. actually work

## The Solution: Textual's Built-in Testing Framework

Textual provides a **Pilot** object that allows programmatic testing of TUI applications. This enables:
- ✅ Simulating real keypresses (ENTER, arrows, typing)
- ✅ Testing user interactions programmatically
- ✅ Verifying apps don't crash on key events
- ✅ Taking screenshots for visual regression testing

## Quick Implementation

### 1. Create a Test File

```python
#!/usr/bin/env python3
"""
Test script for Textual TUI applications
"""

import asyncio
import sys
from pathlib import Path

# Add your app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import your Textual app
from your_app import YourTextualApp

async def test_app_interactions():
    """Test that user interactions work properly"""
    print("🧪 Testing TUI interactions...")
    
    app = YourTextualApp()
    
    try:
        async with app.run_test(size=(80, 25)) as pilot:
            print("✅ App started successfully")
            
            # Wait for app to initialize
            await pilot.pause(1.0)
            
            # Test ENTER key
            print("⌨️  Testing ENTER key...")
            await pilot.press("enter")
            print("✅ ENTER key works!")
            
            # Test arrow keys
            print("⌨️  Testing arrow navigation...")
            await pilot.press("down", "down", "up")
            print("✅ Arrow keys work!")
            
            # Test typing
            print("⌨️  Testing text input...")
            await pilot.press("h", "e", "l", "l", "o")
            print("✅ Text input works!")
            
            # Test hotkeys
            print("⌨️  Testing hotkeys...")
            await pilot.press("escape")
            await pilot.press("ctrl+c")
            print("✅ Hotkeys work!")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    success = await test_app_interactions()
    if success:
        print("\n🎉 All tests passed! TUI interactions work properly.")
    else:
        print("\n💥 Tests failed! Check the errors above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)
```

### 2. Run the Test

```bash
python test_your_app.py
```

## Available Pilot Methods

### Key Simulation
```python
# Single keys
await pilot.press("enter")
await pilot.press("escape")
await pilot.press("space")

# Multiple keys in sequence
await pilot.press("down", "down", "enter")

# Typing text
await pilot.press("h", "e", "l", "l", "o")

# Modifiers
await pilot.press("ctrl+c")
await pilot.press("shift+tab")
await pilot.press("alt+f4")
```

### Mouse Simulation
```python
# Click at coordinates
await pilot.click(10, 5)

# Hover over elements
await pilot.hover(20, 10)
```

### Timing and Synchronization
```python
# Wait for app to process
await pilot.pause(0.5)

# Wait for specific conditions
await pilot.wait_for_scheduled_animations()
```

### Visual Testing
```python
# Take screenshot for visual regression testing
screenshot = pilot.app.export_screenshot()

# Compare with expected output
assert screenshot == expected_screenshot
```

## Common Test Patterns

### Testing Navigation
```python
async def test_navigation():
    async with app.run_test() as pilot:
        # Test moving through a list
        await pilot.press("down", "down", "down")
        
        # Verify cursor position
        table = pilot.app.query_one("#my-table")
        assert table.cursor_row == 3
```

### Testing Form Input
```python
async def test_form_input():
    async with app.run_test() as pilot:
        # Focus on input field
        await pilot.press("tab")
        
        # Type text
        await pilot.press(*"username")
        
        # Submit form
        await pilot.press("enter")
```

### Testing Error Conditions
```python
async def test_error_handling():
    async with app.run_test() as pilot:
        # Trigger an error condition
        await pilot.press("ctrl+x")  # Some error-causing action
        
        # Verify app doesn't crash
        assert pilot.app.is_running
        
        # Verify error message appears
        error_widget = pilot.app.query_one("#error-message")
        assert error_widget.visible
```

## Debugging Tips

### 1. Add Pauses for Debugging
```python
await pilot.pause(2.0)  # Gives time to see what's happening
```

### 2. Print App State
```python
# Print widget states for debugging
table = pilot.app.query_one("#my-table")
print(f"Cursor at row: {table.cursor_row}")
print(f"Selected items: {len(selected_items)}")
```

### 3. Capture Screenshots
```python
# Save screenshot to file for inspection
screenshot = pilot.app.export_screenshot()
with open("debug_screenshot.svg", "w") as f:
    f.write(screenshot)
```

## Integration with pytest

For more comprehensive testing, integrate with pytest:

```python
import pytest
from textual.app import App

@pytest.mark.asyncio
async def test_enter_key_functionality():
    app = MyApp()
    async with app.run_test() as pilot:
        await pilot.press("enter")
        # Add assertions here
        assert some_condition_is_true
```

## When to Use This Approach

Use Textual testing when Claude Code says:
- "I cannot test interactive applications"
- "I cannot press keys in the TUI"
- "I cannot verify the ENTER key works"
- "I can only see escape sequences, not the actual interface"
- "I cannot interact with your Textual application"

## Benefits

- ✅ **Real interaction testing** - Actually simulates user input
- ✅ **Crash detection** - Catches errors that only occur during interaction
- ✅ **Regression testing** - Prevents UI bugs from returning
- ✅ **Visual testing** - Can compare screenshots
- ✅ **CI/CD integration** - Can run in automated testing pipelines

## Quick Reference Commands

```bash
# Create test file
touch test_app.py

# Run test
python test_app.py

# Run with pytest (if using pytest integration)
pytest test_app.py -v

# Install testing dependencies if needed
pip install pytest pytest-asyncio
```

---

**Give this document to Claude Code whenever you're working on Textual TUI applications and it cannot test interactive functionality. This enables proper testing of user interactions, keyboard input, and UI behavior.**