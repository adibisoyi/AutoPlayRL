# hud_monitor.py

import Quartz
import numpy as np
from PIL import ImageGrab
import cv2
import pytesseract
import re
import json
import os

class HUDMonitor:
    def __init__(self, game_window_name="Nestopia"):
        self.game_window_name = game_window_name

    def _get_window_bounds(self):
        options = Quartz.kCGWindowListOptionOnScreenOnly
        window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)

        for window in window_list:
            title = window.get('kCGWindowName', '')
            owner = window.get('kCGWindowOwnerName', '')
            if self.game_window_name in owner or self.game_window_name in title:
                bounds = window['kCGWindowBounds']
                x = int(bounds['X'])
                y = int(bounds['Y'])
                width = int(bounds['Width'])
                height = int(bounds['Height'])
                return (x, y, x + width, y + height)
        return None

    def _preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Invert colors if background is dark
        mean_intensity = np.mean(gray)
        if mean_intensity < 127:
            gray = cv2.bitwise_not(gray)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Resize for better OCR
        scale_factor = 3.0
        resized = cv2.resize(enhanced, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # Blur and sharpen
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

        _, thresholded = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded

    def extract_hud_info(self, debug=False):
        bounds = self._get_window_bounds()
        if bounds is None:
            return None

        x1, y1, x2, y2 = bounds
        width = x2 - x1
        height = y2 - y1

        # Define two horizontal strips: one near top (excluding title bar), one near bottom
        strip_height = max(60, height // 5)  # Increased from 40 and 1/6th of height

        # Skip the window title bar by starting 30 pixels below the top
        top_strip_bbox = (x1, y1 + 30, x2, y1 + 30 + strip_height)
        bottom_strip_bbox = (x1, y2 - 10 - strip_height, x2, y2 - 10)

        def ocr_strip(bbox):
            screenshot = ImageGrab.grab(bbox=bbox)
            img_np = np.array(screenshot)
            processed = self._preprocess_image(img_np)
            padded = cv2.copyMakeBorder(processed, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
            ocr_text = pytesseract.image_to_string(padded, config='--psm 7 -c tessedit_char_whitelist=0123456789').strip()
            # Fallback: try alternate preprocessing if OCR result is empty
            if not ocr_text:
                # Fallback: try with just grayscale and threshold
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                _, fallback_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                fallback_padded = cv2.copyMakeBorder(fallback_thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
                ocr_text = pytesseract.image_to_string(fallback_padded, config='--psm 7 -c tessedit_char_whitelist=0123456789').strip()
            return ocr_text

        top_text = ocr_strip(top_strip_bbox)
        bottom_text = ocr_strip(bottom_strip_bbox)

        # Determine which strip contains more numeric characters
        def numeric_score(text):
            return sum(c.isdigit() for c in text)

        top_score = numeric_score(top_text)
        bottom_score = numeric_score(bottom_text)

        if debug:
            print("[TOP HUD OCR]:", repr(top_text))
            print("[BOTTOM HUD OCR]:", repr(bottom_text))
            print(f"[NUMERIC SCORES] Top: {top_score}, Bottom: {bottom_score}")

        top_digits = re.findall(r'\d+', top_text)
        bottom_digits = re.findall(r'\d+', bottom_text)

        final_text = ''
        if top_score >= bottom_score and top_score > 0:
            final_text = ' '.join(top_digits) if top_digits else top_text
        elif bottom_score > 0:
            final_text = ' '.join(bottom_digits) if bottom_digits else bottom_text
        else:
            fallback = top_text or bottom_text
            final_text = ' '.join(re.findall(r'\d+', fallback)) or fallback

        return {'hud_text': final_text}