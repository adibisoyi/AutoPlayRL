# scripts/agent_utils/screen_monitor.py

import os
import cv2
import numpy as np

# load your templates once
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
TEMPLATE_NAMES = ("game_over1.png", "life_lost.png", "pause.png")
TEMPLATES = []
for name in TEMPLATE_NAMES:
    path = os.path.join(TEMPLATE_DIR, name)
    tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        print(f"[WARN] Failed to load template: {path}")
    TEMPLATES.append(tpl)

def is_special_screen(img_rgb, match_threshold=0.8):
    """
    Detect if any of the static templates appear in the current screen
    via normalized cross-correlation template matching.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    for tpl in TEMPLATES:
        if tpl is None:
            continue
        # perform template matching
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= match_threshold:
            return True
    return False