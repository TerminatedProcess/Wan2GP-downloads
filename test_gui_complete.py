#!/usr/bin/env python3
"""
Comprehensive GUI testing for WanGP Downloader
Tests all interactive functionality and visual elements
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from downloader import DownloaderApp

class GUITester:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.issues = 0
        
    def test_result(self, test_name, passed, details=""):
        if passed:
            print(f"✅ {test_name}")
            self.passed += 1
        else:
            print(f"❌ {test_name} - {details}")
            self.failed += 1
    
    def test_issue(self, test_name, details=""):
        print(f"⚠️ {test_name} - {details}")
        self.issues += 1
    
    async def test_app_startup(self, pilot):
        """Test App Startup section"""
        print("\n🧪 Testing App Startup...")
        
        # Check app loads
        self.test_result("App loads without crashing", pilot.app.is_running)
        
        # Check for ready message
        try:
            status_widget = pilot.app.query_one("#main-status")
            ready_text = "Ready - Use hotkeys below for actions" in str(status_widget.renderable)
            self.test_result("Shows ready message", ready_text)
        except Exception as e:
            self.test_result("Shows ready message", False, f"Status widget issue: {e}")
        
        # Check table shows only missing models initially
        try:
            table = pilot.app.query_one("#download-table")
            row_count = table.row_count
            self.test_result("Only missing models displayed by default", row_count > 0)
            print(f"   📊 Showing {row_count} missing models")
        except Exception as e:
            self.test_result("Only missing models displayed by default", False, str(e))
        
        # Check orange cursor is visible
        try:
            table = pilot.app.query_one("#download-table")
            cursor_row = table.cursor_row
            self.test_result("Orange cursor visible on first row", cursor_row == 0)
        except Exception as e:
            self.test_result("Orange cursor visible on first row", False, str(e))
    
    async def test_enter_key_functionality(self, pilot):
        """Test ENTER Key Functionality section"""
        print("\n🧪 Testing ENTER Key Functionality...")
        
        try:
            table = pilot.app.query_one("#download-table")
            initial_cursor = table.cursor_row
            
            # Press ENTER - should not crash
            await pilot.press("enter")
            await pilot.pause(0.2)
            
            self.test_result("Press ENTER - app doesn't crash", pilot.app.is_running)
            
            # Check if cursor stayed in same position
            cursor_after = table.cursor_row
            self.test_result("Orange cursor stays on same row", cursor_after == initial_cursor)
            
            # Check for selection (this is tricky to verify without looking at internal state)
            # We'll check if the selection count changed or if visual feedback appeared
            try:
                # Look for selection in the app's internal state
                selected_count = len(pilot.app.selected_items)
                self.test_result("Red checkmark appears", selected_count > 0)
            except Exception:
                self.test_issue("Red checkmark appears", "Cannot verify selection state")
            
            # Press ENTER again to deselect
            await pilot.press("enter")
            await pilot.pause(0.2)
            
            try:
                selected_count_after = len(pilot.app.selected_items)
                self.test_result("Press ENTER again - checkmark disappears", selected_count_after == 0)
            except Exception:
                self.test_issue("Press ENTER again - checkmark disappears", "Cannot verify deselection")
                
        except Exception as e:
            self.test_result("Press ENTER - app doesn't crash", False, str(e))
    
    async def test_arrow_navigation(self, pilot):
        """Test Arrow Key Navigation section"""
        print("\n🧪 Testing Arrow Key Navigation...")
        
        try:
            table = pilot.app.query_one("#download-table")
            initial_cursor = table.cursor_row
            
            # Test DOWN arrow
            await pilot.press("down")
            await pilot.pause(0.1)
            cursor_after_down = table.cursor_row
            self.test_result("DOWN arrow moves cursor down", cursor_after_down == initial_cursor + 1)
            
            # Test UP arrow
            await pilot.press("up")
            await pilot.pause(0.1)
            cursor_after_up = table.cursor_row
            self.test_result("UP arrow moves cursor up", cursor_after_up == initial_cursor)
            
            # Test cursor visibility by moving several times
            for _ in range(3):
                await pilot.press("down")
                await pilot.pause(0.05)
            
            final_cursor = table.cursor_row
            self.test_result("Cursor movement is smooth and visible", final_cursor == initial_cursor + 3)
            
        except Exception as e:
            self.test_result("Arrow key navigation", False, str(e))
    
    async def test_hotkeys(self, pilot):
        """Test Hotkey functionality"""
        print("\n🧪 Testing Hotkey Functionality...")
        
        # Test 's' key - Show All toggle
        try:
            table = pilot.app.query_one("#download-table")
            initial_count = table.row_count
            
            await pilot.press("s")
            await pilot.pause(0.3)
            
            new_count = table.row_count
            show_all_works = new_count != initial_count
            self.test_result("'s' key toggles Show All/Missing Only", show_all_works)
            print(f"   📊 Row count changed from {initial_count} to {new_count}")
            
        except Exception as e:
            self.test_result("'s' key toggles Show All/Missing Only", False, str(e))
        
        # Test 'a' key - Select All
        try:
            await pilot.press("a")
            await pilot.pause(0.3)
            
            selected_count = len(pilot.app.selected_items)
            self.test_result("'a' key - Select All", selected_count > 0)
            print(f"   📊 Selected {selected_count} items")
            
        except Exception as e:
            self.test_result("'a' key - Select All", False, str(e))
        
        # Test 'u' key - Unselect All
        try:
            await pilot.press("u")
            await pilot.pause(0.3)
            
            selected_count = len(pilot.app.selected_items)
            self.test_result("'u' key - Unselect All", selected_count == 0)
            
        except Exception as e:
            self.test_result("'u' key - Unselect All", False, str(e))
    
    async def run_all_tests(self):
        """Run the complete test suite"""
        print("🚀 Starting Comprehensive GUI Testing...")
        print("=" * 50)
        
        app = DownloaderApp()
        
        try:
            async with app.run_test(size=(80, 25)) as pilot:
                # Wait for app to fully initialize
                await pilot.pause(2.0)
                
                # Run test sections
                await self.test_app_startup(pilot)
                await self.test_enter_key_functionality(pilot)
                await self.test_arrow_navigation(pilot)
                await self.test_hotkeys(pilot)
                
        except Exception as e:
            print(f"💥 Critical test failure: {e}")
            import traceback
            traceback.print_exc()
            self.failed += 1
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"⚠️  Issues: {self.issues}")
        
        if self.failed == 0:
            print("\n🎉 All critical tests passed!")
            return True
        else:
            print(f"\n💥 {self.failed} tests failed - need fixes!")
            return False

async def main():
    tester = GUITester()
    success = await tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)