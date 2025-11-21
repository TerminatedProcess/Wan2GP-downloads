# WanGP Downloader GUI Test Checklist

## Basic Functionality Tests

### App Startup
- [ ] **App loads without crashing**
- [ ] **Shows "Ready - Use hotkeys below for actions" message**
- [ ] **Only missing models are displayed by default (not existing locally/hub)**
- [ ] **Orange cursor is visible on first row of missing models**
- [ ] **Press 's' to show all models (including existing ones)**

### ENTER Key Functionality  
- [ ] **Press ENTER on first row - app doesn't crash**
- [ ] **Red checkmark (✓) appears in checkbox column**
- [ ] **Orange cursor stays on same row (doesn't jump to first)**
- [ ] **Selection count updates at bottom**
- [ ] **Press ENTER again - red checkmark disappears**

### Arrow Key Navigation
- [ ] **DOWN arrow moves orange cursor down one row**
- [ ] **UP arrow moves orange cursor up one row**
- [ ] **Cursor wraps around at top/bottom of list**
- [ ] **Cursor movement is smooth and visible**

### Mouse Interaction
- [ ] **Click on any row moves orange cursor to that row**
- [ ] **Click does NOT select items (no red checkmark)**
- [ ] **Only ENTER key creates red checkmarks**
- [ ] **Mouse wheel scrolls through the list**

### Hotkey Testing
- [ ] **'a' key - Select All shows red checkmarks on all visible items**
- [ ] **'u' key - Unselect All removes all red checkmarks**
- [ ] **'s' key - Toggle between "Show All" and "Missing Only"**
- [ ] **'r' key - Reset cache (shows reset message)**
- [ ] **'^p' key - Toggle color palette**
- [ ] **ESC key - Exits application cleanly**

### Color Verification
- [ ] **Selected checkmarks are RED (✓)**
- [ ] **Current cursor row has ORANGE highlighting**
- [ ] **Unselected checkboxes are empty/white**
- [ ] **Status text shows correct colors**

### Selection State Persistence
- [ ] **Select multiple items with ENTER**
- [ ] **Navigate with arrows - selections stay red**
- [ ] **Use 's' to toggle view - selections preserved**
- [ ] **Selection count matches visible red checkmarks**

### Edge Cases
- [ ] **Navigate to last item and press DOWN - handles gracefully**
- [ ] **Navigate to first item and press UP - handles gracefully**
- [ ] **Rapid ENTER presses don't cause crashes**
- [ ] **Mix of mouse clicks + keyboard navigation works**

### Visual Layout
- [ ] **All columns are properly aligned**
- [ ] **Text doesn't overflow or get cut off**
- [ ] **Progress info displays correctly at bottom**
- [ ] **Hotkey help bar shows all options**

### Performance
- [ ] **App responds quickly to keypresses**
- [ ] **No visible lag when navigating**
- [ ] **Smooth scrolling through long lists**
- [ ] **Memory usage remains stable**

---

## Test Execution Notes

**Status Legend:**
- ✅ = Passed
- ❌ = Failed  
- ⚠️ = Partial/Issue
- ⏳ = In Progress

**Color Reference:**
- 🔴 **RED** = Selected checkmarks (✓)
- 🟠 **ORANGE** = Current cursor position
- ⚪ **WHITE** = Unselected checkboxes
- 🔵 **BLUE** = Header row

**Critical Tests** (Must Pass):
1. ENTER key doesn't crash
2. Red checkmarks appear/disappear correctly  
3. Orange cursor moves and stays visible
4. Mouse clicks move cursor without selecting

---

*Ready to begin testing when you give the go-ahead!*