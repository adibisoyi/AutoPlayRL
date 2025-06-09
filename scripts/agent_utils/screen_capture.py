import Quartz
from PIL import ImageGrab
import numpy as np

def get_window_bounds_mac(window_name):
    """
    Retrieve the on-screen bounds for a window matching window_name.
    """
    opts = Quartz.kCGWindowListOptionOnScreenOnly
    wins = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID)
    for w in wins:
        name = w.get('kCGWindowName', '')
        owner = w.get('kCGWindowOwnerName', '')
        if window_name in name or window_name in owner:
            b = w['kCGWindowBounds']
            return [
                int(b['X']), int(b['Y']),
                int(b['X'] + b['Width']), int(b['Y'] + b['Height'])
            ]
    return None

def capture_screen(region=None):
    """Capture screen or a region."""
    img = ImageGrab.grab(bbox=region).convert("RGB")
    return np.array(img)

def get_window_region(window_name="Nestopia"):
    return get_window_bounds_mac(window_name)