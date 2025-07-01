#!/usr/bin/env python3
import sys
import os
import time
import argparse
import logging

# 1) Make this scripts/ folder itself a top‐level module
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# 2) Make yolov5/ a top‐level module so its own internal imports (from utils.general) resolve
YOLO_ROOT = os.path.join(THIS_DIR, "yolov5")
if YOLO_ROOT not in sys.path:
    sys.path.insert(0, YOLO_ROOT)

# Now all your existing imports in agent.py and state_extractor.py will work, 
# without changing a single one of them.
from agent import run_agent

# Argument parser setup
parser = argparse.ArgumentParser(description="Launch RL agent")
parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
parser.add_argument("--delay",    type=float, default=0.0, help="Delay between actions (seconds)")
parser.add_argument("--window-title", type=str, default="Nestopia", help="Game window title")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

if __name__ == "__main__":
    #input("[ACTION REQUIRED] Make sure the Nestopia window is visible, then press Enter to start...\n")
    run_agent(episodes=args.episodes, delay=args.delay, window_title=args.window_title)
