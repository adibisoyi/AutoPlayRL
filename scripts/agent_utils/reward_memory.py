import pickle
import os

reward_table = {}
action_usage = {}

def update_reward_table(action_key, reward):
    alpha = 0.2
    clipped = max(-10.0, min(10.0, reward))
    if action_key in reward_table:
        reward_table[action_key] = reward_table[action_key] * (1 - alpha) + clipped * alpha
    else:
        reward_table[action_key] = clipped
    action_usage[action_key] = action_usage.get(action_key, 0) + 1
    # print(f"[DEBUG] Reward '{action_key}' -> {reward_table[action_key]:.2f}")

def save_rewards(path='data/memory.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(reward_table, f)

def load_rewards(path='data/memory.pkl'):
    global reward_table
    if os.path.exists(path):
        with open(path, 'rb') as f:
            reward_table = pickle.load(f)

def get_best_action():
    if not reward_table:
        return None
    sorted_actions = sorted(reward_table.items(), key=lambda x: -x[1])
    return [a for a,_ in sorted_actions[:5]]