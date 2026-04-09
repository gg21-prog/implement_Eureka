import gymnasium as gym
import numpy as np
import traceback
from typing import Tuple


def validate_reward_code(
    reward_code: str,
    env_id: str,
    seed: int = 42,
) -> Tuple[bool, str]:
    """
    Validates LLM-generated reward code in two stages.

    Stage 1 — Compile check:
        exec() the code. Verify compute_reward is defined and callable.

    Stage 2 — Dry-run:
        Create one MuJoCo env (no render, no wrapper).
        Call compute_reward(obs, prev_obs, action, info, env) for DRY_RUN_STEPS steps.
        Verify return value is (numeric, dict) with no NaN/Inf.

    Returns (is_valid: bool, message: str).
    message is "" on success, error description on failure.
    Never raises — all exceptions are caught and returned as failure messages.
    """
    from src.config import DRY_RUN_STEPS

    # ── Stage 1: Compile ──────────────────────────────────────────────────────
    local_ns = {}
    try:
        exec(reward_code, local_ns)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"Compile error: {type(e).__name__}: {e}"

    if "compute_reward" not in local_ns:
        return False, "compute_reward function not defined in generated code"

    reward_fn = local_ns["compute_reward"]

    if not callable(reward_fn):
        return False, "compute_reward is not callable"

    # ── Stage 2: Dry-run ──────────────────────────────────────────────────────
    env = None
    try:
        kwargs = dict(render_mode=None)
        if "Ant" in env_id:
            kwargs["use_contact_forces"] = True
        env = gym.make(env_id, **kwargs)
        obs, info = env.reset(seed=seed)
        prev_obs = obs.copy()
        action = env.action_space.sample()

        for step in range(DRY_RUN_STEPS):
            obs, _, terminated, truncated, info = env.step(action)

            try:
                result = reward_fn(obs, prev_obs, action, info, env)
            except Exception as e:
                return False, (
                    f"compute_reward raised exception at dry-run step {step}: "
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                )

            if not isinstance(result, (tuple, list)) or len(result) != 2:
                return False, (
                    f"compute_reward must return (float, dict), "
                    f"got {type(result)} at step {step}"
                )

            reward_val, components = result

            if not isinstance(reward_val, (int, float, np.floating, np.integer)):
                return False, (
                    f"compute_reward first return value must be numeric, "
                    f"got {type(reward_val)} at step {step}"
                )

            if not isinstance(components, dict):
                return False, (
                    f"compute_reward second return value must be dict, "
                    f"got {type(components)} at step {step}"
                )

            if np.isnan(float(reward_val)) or np.isinf(float(reward_val)):
                return False, f"compute_reward returned NaN or Inf at step {step}"

            prev_obs = obs.copy()
            action = env.action_space.sample()

            if terminated or truncated:
                obs, info = env.reset(seed=seed + step)
                prev_obs = obs.copy()

    except Exception as e:
        return False, f"Dry-run environment error: {type(e).__name__}: {e}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return True, ""
