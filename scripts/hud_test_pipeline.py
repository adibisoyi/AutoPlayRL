from hud_analyser import HUDAnalyser
# hud_test_pipeline.py
import sys
import subprocess
import json
import time
import os
from hud_monitor import HUDMonitor

def bring_nestopia_to_front():
    try:
        script = 'tell application "Nestopia" to activate'
        subprocess.run(["osascript", "-e", script])
        time.sleep(0.5)
    except Exception as e:
        print(f"[ERROR] Could not bring Nestopia to front: {e}")

bring_nestopia_to_front()
#input("[ACTION REQUIRED] Please make sure the game window is visible and press Enter to continue...")

# Step 1: No manual selection needed; HUD will be detected automatically

# Step 3: Read and extract HUD info using selected region
print("[INFO] Extracting HUD info using automatic strip detection...")
hud_monitor = HUDMonitor()
hud_info = hud_monitor.extract_hud_info(debug=True)

analyser = HUDAnalyser()
if hud_info:
    print("\n✅ HUD Info Extracted:")
    for k, v in hud_info.items():
        print(f"  {k}: {v}")
    if 'hud_text' in hud_info:
        analyser.update(hud_info['hud_text'])
        print("\n📊 HUD Slot Analysis:")
        for idx, slot in enumerate(analyser.debug_slot_info()):
            print(f"  Slot {idx}: Value = {slot['value']}, Score = {slot['score']:.2f}, Status = {slot['status']}")
else:
    print("❌ HUD info could not be extracted.")