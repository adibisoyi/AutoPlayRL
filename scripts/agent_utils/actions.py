import logging
import random
import time
import subprocess
from pynput.keyboard import Controller, Key

# Blacklist and failure tracking for zero‐movement actions
_FAILURE_THRESHOLD = 3
_blacklist = set()            # set of tuples of key sequences to never try again
_failure_counts = {}          # maps tuple(key sequence) -> consecutive zero‐movement count

# Track whether an action moved the character.
def record_action_result(keys, moved):
    """
    Track whether an action moved the character.
    If an action yields no movement _FAILURE_THRESHOLD times in a row, blacklist it.
    """
    ktuple = tuple(keys)
    if moved:
        _failure_counts[ktuple] = 0
    else:
        count = _failure_counts.get(ktuple, 0) + 1
        _failure_counts[ktuple] = count
        if count >= _FAILURE_THRESHOLD:
            _blacklist.add(ktuple)
            if list(ktuple) in _action_universe:
                _action_universe.remove(list(ktuple))

ALL_KEYS = [Key.right, Key.alt,Key.left, Key.up, Key.down, Key.shift]
keyboard = Controller()


def set_action_universe(actions):
    """
    Replace the action universe with a new list of key sequences (list of lists of strings).
    Example: [['up'], ['left','A']]
    """
    global _action_universe
    _action_universe = actions

def get_action_universe():
    """
    Return the current action universe (list of key‐sequence lists).
    """
    return _action_universe

def stringify_key(key):
    return key.name if isinstance(key, Key) else str(key)

def parse_key(key_str):
    return getattr(Key, key_str) if hasattr(Key, key_str) else key_str

# 1) build a list of basic single‐key actions (string form)
BASIC_ACTIONS = [[stringify_key(k)] for k in ALL_KEYS]
# 2) module‐level action universe, initially only basic actions
_action_universe = BASIC_ACTIONS.copy()

def bring_nestopia_to_front():
    try:
        subprocess.run(["osascript", "-e", 'tell application "Nestopia" to activate'])
        time.sleep(0.05)
    except Exception as e:
        logging.error("Could not focus Nestopia: %s", e)

def random_key_combination(max_keys=2):
    # choose from dynamic universe if available
    if _action_universe:
        # only choose from non-blacklisted actions
        valid = [a for a in _action_universe if tuple(a) not in _blacklist]
        if valid:
            return random.choice(valid)
        # if everything is blacklisted, fall back to full universe
        return random.choice(_action_universe)
    # fallback: random sample from ALL_KEYS
    k = random.randint(1, max_keys)
    return [stringify_key(k_) for k_ in random.sample(ALL_KEYS, k)]

def perform_action(keys, duration=0.1, verbose=True):
    # ensure Nestopia is focused and ready
    bring_nestopia_to_front()
    # brief pause to let window come to front
    time.sleep(0.1)
    for k in keys:
        try:
            keyboard.press(parse_key(k))
        except:
            pass
    time.sleep(duration)
    for k in keys:
        try:
            keyboard.release(parse_key(k))
        except:
            pass
    # sync: brief pause to allow frame/log processing
    time.sleep(0.1)
    if verbose:
        logging.info("[ACTION] %s for %.2fs", '+'.join(keys), duration)
    return keys
