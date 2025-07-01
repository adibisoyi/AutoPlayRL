# scripts/agent_utils/reward_model.py

import numpy as np
import os
import logging

# Optionally squash reward using tanh, controlled by environment variable
USE_REWARD_TANH = os.getenv("USE_REWARD_TANH", "0") == "1"

def _count(x):
    """Return numeric value or length for iterables, else 0."""
    if isinstance(x, (int, float)):
        return x
    try:
        return len(x)
    except Exception:
        return 0

# Reward model configuration
HORIZONTAL_WEIGHT = 1.0
STAGNATION_PENALTY = 0.02
VERTICAL_WEIGHT = 0.3
DOWNWARD_PENALTY = 0.2
HUD_DIVISOR = 10.0
LIFE_LOSS_PENALTY = 1.0
ENEMY_AVOID_WEIGHT = 0.5      # reward for increasing distance from nearest enemy
ENEMY_KILL_WEIGHT = 0.6      # reward per enemy defeated
COIN_WEIGHT = 0.3            # reward per coin collected
POWERUP_WEIGHT = 0.7         # reward per power-up collected

# Enemy proximity configuration
ENEMY_PROXIMITY_THRESHOLD = float(os.getenv("ENEMY_PROXIMITY_THRESHOLD", "50.0"))  # px
CLOSE_PENALTY_WEIGHT = 0.2  # penalty when enemy is too close

# Generic gameplay event weights (not game-specific)
PROGRESSION_WEIGHT = 1.0        # reward for level or mission progress
ITEM_COLLECTION_WEIGHT = 0.5    # reward for picking up items or collectibles
COMBO_BONUS = 0.2               # bonus for chaining actions or combos
TIME_PENALTY = 0.01             # small penalty to encourage efficient play

def calculate_reward(prev_state, next_state, hud_before, hud_after, hud_analyser, screen_shape, dx, dy):
    reward = 0.0
    # movement debug
    logging.debug("Movement dx=%s, dy=%s", dx, dy)

    # 1) Horizontal progress (normalized)
    p0 = prev_state.get("player_pos")
    n0 = next_state.get("player_pos")
    if p0 is not None and n0 is not None:
        dx = n0[0] - p0[0]
        reward += HORIZONTAL_WEIGHT * dx / screen_shape[1]
        # 1b) Stagnation penalty: discourage no horizontal progress
        if dx == 0:
            reward -= STAGNATION_PENALTY

        # 1c) Vertical progress: normalized upward movement
        # (assumes origin at top-left; moving up decreases the y-coordinate)
        py, ny = p0[1], n0[1]
        vy = (py - ny) / screen_shape[0]  # upward positive
        # weight vertical less than horizontal
        reward += VERTICAL_WEIGHT * vy
        # discourage downward movement
        if vy < 0:
            reward += vy * DOWNWARD_PENALTY

    # 2) HUD contributions: raw digit change + weighted delta
    prev_tokens = hud_before.get("hud_text", "").split()
    curr_tokens = hud_after.get("hud_text", "").split()
    # raw HUD delta: sum of all numeric token changes
    raw_before = sum(int(t) for t in prev_tokens if t.isdigit())
    raw_after  = sum(int(t) for t in curr_tokens if t.isdigit())
    raw_hud_delta = raw_after - raw_before
    raw_hud_delta = max(min(raw_hud_delta, 100), -100)  # Clamp extreme values
    # weighted HUD delta from analyser
    weighted_hud_delta = hud_analyser.get_reward_delta(prev_tokens, curr_tokens)
    # smooth and normalize HUD contributions
    # raw HUD: compress large jumps via tanh, safer scaling
    raw_scaled = np.tanh(raw_hud_delta / HUD_DIVISOR)
    # weighted HUD: also compress via tanh
    weighted_scaled = np.tanh(weighted_hud_delta)
    reward += raw_scaled + weighted_scaled
    logging.debug(
        "[REWARD DEBUG] raw=%+.2f -> scaled=%+.2f, weighted=%+.2f -> scaled=%+.2f, combined=%+.2f, total_reward=%+.2f",
        raw_hud_delta,
        raw_scaled,
        weighted_hud_delta,
        weighted_scaled,
        raw_scaled + weighted_scaled,
        reward,
    )

    # 3) Enemy handling: reward defeating or avoiding enemies
    # 3a) Reward for defeating enemies (reducing enemy count)
    prev_enemies_list = prev_state.get("enemies", [])
    next_enemies_list = next_state.get("enemies", [])
    defeated = max(_count(prev_enemies_list) - _count(next_enemies_list), 0)
    reward += ENEMY_KILL_WEIGHT * defeated
    # 3b) Avoidance and proximity penalty
    prev_positions = prev_state.get("enemy_positions", [])
    next_positions = next_state.get("enemy_positions", [])
    if p0 is not None and prev_positions and next_positions and defeated == 0:
        # compute min distance to enemy before and after
        prev_dists = [np.hypot(p0[0]-ex, p0[1]-ey) for ex, ey in prev_positions]
        next_dists = [np.hypot(n0[0]-ex, n0[1]-ey) for ex, ey in next_positions]
        min_prev, min_next = min(prev_dists), min(next_dists)
        # penalty if too close initially
        if min_prev < ENEMY_PROXIMITY_THRESHOLD:
            penalty = (ENEMY_PROXIMITY_THRESHOLD - min_prev) / ENEMY_PROXIMITY_THRESHOLD
            reward -= CLOSE_PENALTY_WEIGHT * penalty
            # reward for moving away if stepping back
            delta = min_next - min_prev
            if delta > 0:
                reward += ENEMY_AVOID_WEIGHT * delta

    # 4) Explicit coin and power-up collection
    coins_before = _count(prev_state.get("coins", 0))
    coins_after  = _count(next_state.get("coins", 0))
    delta_coins  = max(coins_after - coins_before, 0)
    reward += COIN_WEIGHT * delta_coins

    powerups_before = _count(prev_state.get("powerups", 0))
    powerups_after  = _count(next_state.get("powerups", 0))
    delta_powerups = max(powerups_after - powerups_before, 0)
    reward += POWERUP_WEIGHT * delta_powerups

    # Life loss penalty: heavily penalize losing a life (first HUD token assumed lives)
    try:
        if prev_tokens and curr_tokens:
            prev_lives = int(prev_tokens[0]) if prev_tokens[0].isdigit() else None
            curr_lives = int(curr_tokens[0])  if curr_tokens[0].isdigit()  else None
            if prev_lives is not None and curr_lives is not None and curr_lives < prev_lives:
                reward -= LIFE_LOSS_PENALTY * 5
    except Exception:
        pass

    # 5) Generic gameplay events
    prog_before = prev_state.get("level_progress", 0)
    prog_after  = next_state.get("level_progress", 0)
    delta_prog  = prog_after - prog_before
    reward += PROGRESSION_WEIGHT * delta_prog

    items_before = prev_state.get("items_collected", 0)
    items_after  = next_state.get("items_collected", 0)
    delta_items  = items_after - items_before
    reward += ITEM_COLLECTION_WEIGHT * delta_items

    combo_before = prev_state.get("combo_count", 0)
    combo_after  = next_state.get("combo_count", 0)
    delta_combo  = max(combo_after - combo_before, 0)
    reward += COMBO_BONUS * delta_combo

    # time penalty to encourage faster completion
    reward -= TIME_PENALTY

    # optionally squash reward into a bounded range
    if USE_REWARD_TANH:
        reward = np.tanh(reward)
    return reward
