import numpy as np
from collections import deque

class HUDAnalyser:
    def __init__(self, history_length=10, slot_names=None):
        self.history = deque(maxlen=history_length)
        self.slot_names = slot_names
        # If provided, slot_names defines semantic names for each HUD slot
        self.expected_len = None
        self.slots_info = {}
        self._last_tokens = None

    def update(self, hud_tokens):
        # Skip repeated token sequences to avoid redundant processing
        if hud_tokens == self._last_tokens:
            return
        self._last_tokens = list(hud_tokens)

        # Extract only digit tokens
        nums = [int(t) for t in hud_tokens if t.isdigit()]
        if not nums:
            return

        # On first valid call, fix the expected length
        if self.expected_len is None:
            if self.slot_names is not None:
                self.expected_len = len(self.slot_names)
            else:
                self.expected_len = len(nums)
        # If subsequent lengths differ, skip this update
        elif len(nums) != self.expected_len:
            return

        self.history.append(nums)
        self._analyze()

    def _analyze(self):
        # Need at least two consistent entries
        if len(self.history) < 2:
            return

        arr = np.array(self.history)  # Now guaranteed a clean 2D numeric array
        deltas = np.diff(arr, axis=0)

        new_info = {}
        for idx in range(arr.shape[1]):
            key = self.slot_names[idx] if self.slot_names else idx
            changes = deltas[:, idx]
            pos = np.sum(changes > 0)
            neg = np.sum(changes < 0)
            direction = 1 if pos >= neg else -1
            # Weight is signed change frequency
            weight = (pos - neg) / len(changes) if len(changes) else 0.0
            new_info[key] = {"direction": direction, "weight": weight}

        self.slots_info = new_info

    def get_reward_delta(self, prev_tokens, curr_tokens):
        try:
            prev = [int(t) for t in prev_tokens if t.isdigit()]
            curr = [int(t) for t in curr_tokens if t.isdigit()]
        except:
            return 0.0

        reward = 0.0
        for i, (p, c) in enumerate(zip(prev, curr)):
            key = self.slot_names[i] if self.slot_names else i
            info = self.slots_info.get(key, {"direction": 1, "weight": 0.0})
            delta = c - p
            reward += info["direction"] * info["weight"] * delta
        return reward

    def debug_slot_info(self):
        # Returns a dict keyed by slot names (if provided) or indices
        return self.slots_info