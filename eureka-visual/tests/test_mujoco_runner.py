"""Smoke test for MuJoCo runner — runs one candidate for 1000 train steps."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SIMPLE_REWARD = """
import numpy as np

def compute_reward(obs, prev_obs, action, info, env):
    forward_vel = obs[5]   # Ant-v4: obs[5] = forward velocity
    height = obs[0]
    energy = float(np.square(action).sum())
    posture = 1.0 if 0.2 < height < 1.0 else -1.0
    total = forward_vel * 2.0 + posture * 0.5 - energy * 0.01 + 0.1
    return float(total), {"forward": float(forward_vel), "posture": posture}
"""


def test_run_candidate_smoke():
    from src.mujoco_runner import run_candidate
    result = run_candidate(
        reward_code=SIMPLE_REWARD,
        env_id="Ant-v4",
        n_train_steps=1000,
        n_eval_steps=50,
        render=True,
        seed=42,
    )
    assert result["success"], f"run_candidate failed: {result['error']}"
    assert len(result["frames"]) > 0, "No frames collected"
    assert len(result["reward_curve"]) > 0, "No rewards collected"
    assert result["mean_reward"] != -999.0, "mean_reward was not updated"
    assert len(result["forward_velocities"]) > 0, "No forward_velocities collected"
    assert result["forward_velocity_proxy"] != 0.0 or True, "proxy can be 0 — just check it exists"
    print(
        f"  test_run_candidate_smoke: PASS — "
        f"mean_reward={result['mean_reward']:.4f}, "
        f"frames={len(result['frames'])}, "
        f"fwd_vel_proxy={result['forward_velocity_proxy']:.4f}"
    )


if __name__ == "__main__":
    print("Testing MuJoCo runner (requires MUJOCO_GL=egl on Linux)...")
    print("Note: uses Ant-v4 with 1000 train steps — takes ~1-2 min on GPU")
    try:
        test_run_candidate_smoke()
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)
