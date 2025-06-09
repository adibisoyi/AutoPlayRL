import random
import os
from .reward_memory import reward_table
from .actions import set_action_universe
from .actions import random_key_combination
import numpy as np
from itertools import combinations
SOFTMAX_TEMPERATURE = float(os.getenv("SOFTMAX_TEMPERATURE", "1.0"))
USE_SOFTMAX = os.getenv("USE_SOFTMAX", "1") == "1"
MAX_COMBO_KEYS = int(os.getenv("MAX_COMBO_KEYS", "2"))

# dynamic action generation settings
BASIC_KEYS = ["up", "down", "left", "right", "shift", "alt"]
BASIC_PHASE_EPISODES = 100

FORWARD_BIAS = 0.05

epsilon = 0.9
bad_actions = set()

def choose_action(ep=None, max_keys=MAX_COMBO_KEYS, bad_action_streak=0):

    # initialize or update dynamic action universe
    if ep is not None:
        if ep == 1:
            # start with only basic single-key actions
            set_action_universe([[k] for k in BASIC_KEYS])
        elif ep == BASIC_PHASE_EPISODES + 1:
            # after basic phase, expand to all combinations up to max_keys
            all_actions = [
                list(c)
                for r in range(1, max_keys + 1)
                for c in combinations(BASIC_KEYS, r)
            ]
            set_action_universe(all_actions)

    # prune individual keys with negligible effect every 10 episodes
    if ep is not None and ep % 10 == 0:
        for act, rew in list(reward_table.items()):
            if '+' not in act and abs(rew) < 0.01:
                bad_actions.add(act)

    # occasionally pick the historically best action to avoid stagnation
    if reward_table and random.random() < FORWARD_BIAS:
        best = max(reward_table.items(), key=lambda kv: kv[1])[0]
        return best.split('+')

    allow_bad = bad_action_streak >= 4
    action = None
    while action is None or (not allow_bad and '+'.join(action) in bad_actions):
        if random.random() < epsilon or not reward_table:
            action = random_key_combination(MAX_COMBO_KEYS)
        else:
            # exploitation: choose among positive actions
            good = [k for k,v in reward_table.items() if v > 0 and k not in bad_actions]
            if good:
                if USE_SOFTMAX:
                    # softmax sampling over rewards
                    rewards = np.array([reward_table[k] for k in good], dtype=float)
                    exps = np.exp(rewards / SOFTMAX_TEMPERATURE)
                    probs = exps / np.sum(exps)
                    chosen = random.choices(good, weights=probs, k=1)[0]
                else:
                    # original weighted sampling
                    weights = []
                    for k in good:
                        w = reward_table[k]
                        if 'up' in k:
                            w *= 1.1
                        weights.append(w)
                    chosen = random.choices(good, weights=weights, k=1)[0]
                action = chosen.split('+')
            else:
                action = random_key_combination(MAX_COMBO_KEYS)

        if ep is not None and ep > 300 and set(action) <= {"up","down","left","shift"}:
            action = random_key_combination(max_keys)

    return action

def decay_epsilon():
    global epsilon
    epsilon = max(0.01, epsilon * 0.995)


def get_action_duration(action):
    """
    Compute a duration for the given action sequence.
    Uses ACTION_DURATION env var (seconds per key) or defaults to 0.01.
    """
    base = float(os.getenv("ACTION_DURATION", "1.0"))
    return base * len(action)