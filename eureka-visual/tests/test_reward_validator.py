"""Tests for reward code validation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reward_validator import validate_reward_code

VALID_ANT_REWARD = """
import numpy as np

def compute_reward(obs, prev_obs, action, info, env):
    forward_vel = obs[5]
    height = obs[0]
    energy = float(np.square(action).sum())
    posture = 1.0 if 0.2 < height < 1.0 else -1.0
    total = forward_vel * 2.0 + posture * 0.5 - energy * 0.01
    components = {
        "forward_vel": float(forward_vel),
        "posture": posture,
        "energy_penalty": -energy * 0.01,
    }
    return float(total), components
"""

INVALID_SYNTAX = "def compute_reward(obs prev_obs action info env):\n    pass"
MISSING_FUNCTION = "x = 1 + 1"
WRONG_RETURN_TYPE = """
def compute_reward(obs, prev_obs, action, info, env):
    return 1.0  # missing dict — returns a bare float, not a 2-tuple
"""
NAN_RETURN = """
import numpy as np
def compute_reward(obs, prev_obs, action, info, env):
    return float('nan'), {}
"""


def test_valid():
    ok, msg = validate_reward_code(VALID_ANT_REWARD, "Ant-v4")
    assert ok, f"Expected valid, got: {msg}"
    print("  test_valid: PASS")


def test_invalid_syntax():
    ok, msg = validate_reward_code(INVALID_SYNTAX, "Ant-v4")
    assert not ok and "Syntax" in msg, f"Expected SyntaxError, got: {msg}"
    print("  test_invalid_syntax: PASS")


def test_missing_function():
    ok, msg = validate_reward_code(MISSING_FUNCTION, "Ant-v4")
    assert not ok, f"Expected invalid, got valid"
    print("  test_missing_function: PASS")


def test_wrong_return():
    ok, msg = validate_reward_code(WRONG_RETURN_TYPE, "Ant-v4")
    assert not ok, f"Expected invalid return type caught, got valid"
    print("  test_wrong_return: PASS")


def test_nan_return():
    ok, msg = validate_reward_code(NAN_RETURN, "Ant-v4")
    assert not ok and "NaN" in msg, f"Expected NaN caught, got: {msg}"
    print("  test_nan_return: PASS")


if __name__ == "__main__":
    failed = 0
    for t in [test_valid, test_invalid_syntax, test_missing_function,
              test_wrong_return, test_nan_return]:
        try:
            t()
        except Exception as e:
            print(f"  {t.__name__}: FAIL — {e}")
            failed += 1
    print(f"\n{5 - failed}/5 tests passed.")
    sys.exit(0 if failed == 0 else 1)
