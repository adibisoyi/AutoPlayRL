import os
import torch
import numpy as np
import cv2
import logging
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression
from yolov5.models.common import DetectMultiBackend
import time

# Frame skipping for performance
_frame_count = 0
_last_state = None
SKIP_N_FRAMES = int(os.getenv("SKIP_N_FRAMES", "1"))  # skip this many frames between full detections

# Default detection thresholds (can be overridden via env vars or function args)
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.25))
IOU_THRESH  = float(os.getenv("IOU_THRESH", 0.45))
# Default mapping from state keys to object labels
DEFAULT_CLASS_MAPPING = {
    "player": ["mario", "player"],
    "enemy": ["enemy", "goomba", "koopa"],
    "coin": ["coin"],
    "powerup": ["powerup", "mushroom", "flower"],
}
# Allow overriding the model path via environment
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "../../models/best.pt")
)
# Allow dynamic device selection via environment
DEVICE = os.getenv("DEVICE", "cpu")
model = DetectMultiBackend(MODEL_PATH, device=DEVICE)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0]/img0_shape[0], img1_shape[1]/img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1]*gain)/2,
               (img1_shape[0] - img0_shape[0]*gain)/2)
    else:
        gain, pad = ratio_pad
    # Shift by padding
    coords[:, [0,2]] -= pad[0]
    coords[:, [1,3]] -= pad[1]
    # Scale using separate width/height gains if gain is a tuple
    if isinstance(gain, (tuple, list)):
        gain_x, gain_y = gain
    else:
        gain_x = gain_y = gain
    coords[:, [0,2]] /= gain_x
    coords[:, [1,3]] /= gain_y
    # Clamp to image bounds
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

def get_game_state(
    image: np.ndarray,
    class_mapping: dict = None,
    conf_thresh: float = CONF_THRESH,
    iou_thresh: float   = IOU_THRESH
) -> dict:
    global _frame_count, _last_state
    _frame_count += 1
    # If skipping frames and we have a cached state, return it
    if SKIP_N_FRAMES and _frame_count % (SKIP_N_FRAMES + 1) != 0 and _last_state is not None:
        return _last_state
    # Prepare mapping
    mapping = class_mapping if class_mapping is not None else DEFAULT_CLASS_MAPPING
    # Letterbox with returned ratio and padding
    img, ratio, pad = letterbox(image, new_shape=320)
    img = img[:, :, ::-1].transpose(2,0,1)
    img = np.ascontiguousarray(img)
    tensor = torch.from_numpy(img).to(model.device).float() / 255.0
    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)

    pred = model(tensor)[0]
    pred = non_max_suppression(
        pred,
        conf_thres=conf_thresh,
        iou_thres=iou_thresh
    )[0]
    names = model.names

    # Initialize state collections
    state = { key+"_pos" if key=="player" else key+"s":
              None if key=="player" else []
              for key in mapping }
    # Temporary holder for player
    player = None
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(
            tensor.shape[2:], pred[:, :4], image.shape,
            ratio_pad=(ratio, pad)
        ).round()
        for *xyxy, conf, cls in pred:
            x = float((xyxy[0]+xyxy[2]) / 2)
            y = float((xyxy[1]+xyxy[3]) / 2)
            lbl = names[int(cls)].lower()
            # Assign detection to the first matching state key
            for key, labels in mapping.items():
                if lbl in labels:
                    if key == "player":
                        player = (x, y)
                    else:
                        state[key+"s"].append((x, y))
                    break

    # Fill player position and coordinates
    if player is not None:
        state["player_pos"] = player
        state["player_x"], state["player_y"] = player
    else:
        state["player_pos"] = None
        state["player_x"], state["player_y"] = 0, 0
    _last_state = state
    return state

def get_player_movement(capture_fn) -> tuple:
    """
    capture_fn should be a function returning the current screen image (np.ndarray).
    """
    frame1 = capture_fn()
    state1 = get_game_state(frame1)
    time.sleep(0.01)
    frame2 = capture_fn()
    state2 = get_game_state(frame2)

    dx = state2["player_x"] - state1["player_x"]
    dy = state2["player_y"] - state1["player_y"]
    logging.debug("Movement dx=%s, dy=%s", dx, dy)
    return dx, dy
