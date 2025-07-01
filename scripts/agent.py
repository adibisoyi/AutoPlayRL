import time
import os
import logging
import agent_utils.policy as policy
from agent_utils.actions import perform_action
from agent_utils.reward_memory import update_reward_table, save_rewards, load_rewards
from agent_utils.reward_memory import reward_table
from agent_utils.state_extractor import get_game_state
from agent_utils.screen_capture import get_window_region, capture_screen
from hud_monitor import HUDMonitor
from agent_utils.hud_analyser import HUDAnalyser
from agent_utils.reward_model import calculate_reward
from agent_utils.screen_monitor import is_special_screen
from agent_utils.actions import bring_nestopia_to_front


def run_agent(episodes=500, delay=0.0, window_title="Nestopia"):
    load_rewards()

    # track consecutive zero-motion failures to blacklist ineffective actions
    failure_counts = {}
    BLACKLIST_THRESHOLD = 3
    blacklisted_actions = set()

    try:
        region = get_window_region(window_title)
        if region is None:
            raise RuntimeError("Could not locate the game window.")

        # Ensure Nestopia is focused once at the start of training
        bring_nestopia_to_front()
    except Exception as e:
        logging.error("Window capture failed: %s", e)
        return

    hud_monitor = HUDMonitor()
    hud_analyser = HUDAnalyser()
    os.makedirs("logs", exist_ok=True)

    for ep in range(episodes):
        img = capture_screen(region)
        while is_special_screen(img):
                time.sleep(0.01)
                img = capture_screen(region)

        prev_img   = img
        screen_shape = prev_img.shape[:2]
        prev_state = get_game_state(prev_img)
        hud_before = hud_monitor.extract_hud_info()

        # choose an action, skipping any blacklisted combos
        action = policy.choose_action(ep=ep)
        while '+'.join(action) in blacklisted_actions:
            action = policy.choose_action(ep=ep)

        # Use a generic action duration from policy
        duration = policy.get_action_duration(action)
        perform_action(action, duration)

        # Ensure the action has time to take effect and capture after update
        time.sleep(duration)
        start_time = time.time()
        next_img = capture_screen(region)
        next_state = get_game_state(next_img)
        # Poll until state changes or timeout
        while next_state == prev_state and time.time() - start_time < duration + 0.1:
            time.sleep(0.01)
            next_img = capture_screen(region)
            next_state = get_game_state(next_img)
        # compute frame-to-frame player displacement
        dx = next_state.get("player_x", 0) - prev_state.get("player_x", 0)
        dy = next_state.get("player_y", 0) - prev_state.get("player_y", 0)
        logging.debug("Movement dx=%s, dy=%s", dx, dy)
        hud_after = hud_monitor.extract_hud_info()

        # Update HUD analyser history
        hud_analyser.update(hud_before.get("hud_text", "").split())
        hud_analyser.update(hud_after.get("hud_text", "").split())

        action_key = '+'.join(action)
        if dx == 0 and dy == 0:
            failure_counts[action_key] = failure_counts.get(action_key, 0) + 1
            if failure_counts[action_key] >= BLACKLIST_THRESHOLD:
                blacklisted_actions.add(action_key)
                logging.info("Blacklisting action %s after %d zero-motion tries", action_key, failure_counts[action_key])
        else:
            failure_counts[action_key] = 0

        reward = calculate_reward(prev_state, next_state, hud_before, hud_after, hud_analyser, screen_shape, dx, dy)
        update_reward_table('+'.join(action), reward)
        policy.decay_epsilon()

        logging.info("[EP %03d] Action: %s | Reward: %+0.2f | Epsilon: %.2f", ep, '+'.join(action), reward, policy.epsilon)

        if ep % 25 == 0:
            save_rewards()

        time.sleep(delay)

    save_rewards()
    # After training, display top 10 learned actions
    sorted_actions = sorted(reward_table.items(), key=lambda kv: kv[1], reverse=True)
    logging.info("\n[RESULT] Top 10 actions by average reward:")
    for action_key, value in sorted_actions[:10]:
        logging.info("  %s: %.2f", action_key, value)
    logging.info("Training completed.")
