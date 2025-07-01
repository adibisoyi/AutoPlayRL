from hud_analyser import HUDAnalyser
# hud_test_pipeline.py
import sys
import subprocess
import json
import time
import os
from hud_monitor import HUDMonitor
import logging

def bring_nestopia_to_front():
    try:
        script = 'tell application "Nestopia" to activate'
        subprocess.run(["osascript", "-e", script])
        time.sleep(0.5)
    except Exception as e:
        logging.error("Could not bring Nestopia to front: %s", e)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
bring_nestopia_to_front()
#input("[ACTION REQUIRED] Please make sure the game window is visible and press Enter to continue...")

# Step 1: No manual selection needed; HUD will be detected automatically

# Step 3: Read and extract HUD info using selected region
logging.info("Extracting HUD info using automatic strip detection...")
hud_monitor = HUDMonitor()
hud_info = hud_monitor.extract_hud_info(debug=True)

analyser = HUDAnalyser()
if hud_info:
    logging.info("\n HUD Info Extracted:")
    for k, v in hud_info.items():
        logging.info("  %s: %s", k, v)
    if 'hud_text' in hud_info:
        analyser.update(hud_info['hud_text'])
        logging.info("\n HUD Slot Analysis:")
        for idx, slot in enumerate(analyser.debug_slot_info()):
            logging.info("  Slot %d: Value = %s, Score = %.2f, Status = %s", idx, slot['value'], slot['score'], slot['status'])
else:
    logging.error("HUD info could not be extracted.")
