#!/usr/bin/env python3
"""
Test script to verify ENTER key functionality in the downloader app
Uses Textual's built-in testing capabilities
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path so we can import the app
sys.path.insert(0, str(Path(__file__).parent))

from downloader import DownloaderApp

async def test_enter_key():
    """Test that pressing ENTER doesn't crash the app"""
    print("🧪 Testing ENTER key functionality...")
    
    app = DownloaderApp()
    
    try:
        async with app.run_test(size=(80, 25)) as pilot:
            print("✅ App started successfully")
            
            # Wait a moment for app to fully initialize
            await pilot.pause(1.0)
            
            # Simulate pressing ENTER
            print("⌨️  Pressing ENTER key...")
            await pilot.press("enter")
            
            print("✅ ENTER key pressed without crashing!")
            
            # Try pressing it a few more times
            await pilot.press("enter")
            await pilot.press("enter")
            
            print("✅ Multiple ENTER presses successful!")
            
            # Try arrow keys + ENTER
            await pilot.press("down", "down", "enter")
            print("✅ Arrow navigation + ENTER successful!")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def main():
    success = await test_enter_key()
    if success:
        print("\n🎉 All tests passed! ENTER key is working properly.")
    else:
        print("\n💥 Tests failed! ENTER key still has issues.")
    
    return 0 if success else 1

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)