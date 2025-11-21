#!/usr/bin/env python3
"""
Test mouse vs keyboard behavior
- Mouse clicks should only move orange cursor
- ENTER key should toggle red checkmarks
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from downloader import DownloaderApp

async def test_mouse_vs_keyboard():
    """Test that mouse and keyboard have different behaviors"""
    print("🧪 Testing Mouse vs Keyboard Behavior...")
    
    app = DownloaderApp()
    
    try:
        async with app.run_test(size=(80, 25)) as pilot:
            await pilot.pause(2.0)  # Let app initialize
            
            table = pilot.app.query_one("#download-table")
            
            print(f"📍 Starting cursor position: {table.cursor_row}")
            print(f"🔴 Starting selection count: {len(pilot.app.selected_items)}")
            
            # Test 1: Mouse click should move cursor but NOT select
            print("\n🖱️  Testing mouse click behavior...")
            # Click on the table widget at a specific row
            table_widget = pilot.app.query_one("#download-table")
            await pilot.click(table_widget, offset=(10, 3))  # Click on a row
            await pilot.pause(0.2)
            
            cursor_after_click = table.cursor_row
            selections_after_click = len(pilot.app.selected_items)
            
            print(f"📍 Cursor after click: {cursor_after_click}")
            print(f"🔴 Selections after click: {selections_after_click}")
            
            # Mouse click should move cursor but not add selections
            cursor_moved = cursor_after_click != 0
            no_selections_from_click = selections_after_click == 0
            
            print(f"✅ Mouse moved cursor: {cursor_moved}")
            print(f"✅ Mouse did NOT select: {no_selections_from_click}")
            
            # Test 2: ENTER key should toggle selection
            print("\n⌨️  Testing ENTER key behavior...")
            await pilot.press("enter")
            await pilot.pause(0.2)
            
            cursor_after_enter = table.cursor_row
            selections_after_enter = len(pilot.app.selected_items)
            
            print(f"📍 Cursor after ENTER: {cursor_after_enter}")
            print(f"🔴 Selections after ENTER: {selections_after_enter}")
            
            # ENTER should keep cursor same but toggle selection
            cursor_unchanged = cursor_after_enter == cursor_after_click
            selection_toggled = selections_after_enter > selections_after_click
            
            print(f"✅ ENTER kept cursor position: {cursor_unchanged}")
            print(f"✅ ENTER toggled selection: {selection_toggled}")
            
            # Test 3: Multiple clicks should only move cursor
            print("\n🖱️  Testing multiple mouse clicks...")
            initial_selections = len(pilot.app.selected_items)
            
            await pilot.click(10, 10)  # Click different row
            await pilot.pause(0.1)
            await pilot.click(10, 12)  # Click another row
            await pilot.pause(0.1)
            
            final_cursor = table.cursor_row
            final_selections = len(pilot.app.selected_items)
            
            print(f"📍 Final cursor position: {final_cursor}")
            print(f"🔴 Final selection count: {final_selections}")
            
            multiple_clicks_no_select = final_selections == initial_selections
            print(f"✅ Multiple clicks didn't change selections: {multiple_clicks_no_select}")
            
            # Summary
            print("\n📊 BEHAVIOR TEST RESULTS:")
            if cursor_moved and no_selections_from_click:
                print("✅ Mouse clicks: Move cursor only ✓")
            else:
                print("❌ Mouse clicks: Should move cursor, not select")
                
            if cursor_unchanged and selection_toggled:
                print("✅ ENTER key: Toggle selection only ✓") 
            else:
                print("❌ ENTER key: Should toggle selection without moving cursor")
                
            if multiple_clicks_no_select:
                print("✅ Multiple clicks: Only cursor movement ✓")
            else:
                print("❌ Multiple clicks: Should not affect selections")
            
            success = cursor_moved and no_selections_from_click and cursor_unchanged and selection_toggled and multiple_clicks_no_select
            
            if success:
                print("\n🎉 Mouse vs Keyboard behavior is CORRECT!")
                return True
            else:
                print("\n💥 Mouse vs Keyboard behavior needs fixes!")
                return False
                
    except Exception as e:
        print(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    success = await test_mouse_vs_keyboard()
    return 0 if success else 1

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)