import gymnasium as gym
import numpy as np
import traceback

from src.config import N_ENVS, N_TRAIN_STEPS, N_EVAL_STEPS, FRAME_SUBSAMPLE


# ── RewardWrapper ─────────────────────────────────────────────────────────────

class RewardWrapper(gym.Wrapper):
    """
    Wraps a MuJoCo env to use a dynamically injected reward function.

    Injected function signature:
        def compute_reward(obs, prev_obs, action, info, env) -> tuple[float, dict]

    On error: returns (0.0, {"_error": str(e)}) — does NOT suppress silently.
    prev_obs is None-equivalent (zeros) on the first step after reset.
    action is the action that was just applied before obs was returned.
    """

    def __init__(self, env: gym.Env, reward_fn: callable):
        super().__init__(env)
        self._reward_fn = reward_fn
        self._prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs.copy()
        return obs, info

    def step(self, action):
        obs, _orig_reward, terminated, truncated, info = self.env.step(action)
        try:
            reward, components = self._reward_fn(
                obs, self._prev_obs, action, info, self.env
            )
            reward = float(reward)
        except Exception as e:
            reward = 0.0
            components = {"_error": str(e)}
        info["reward_components"] = components
        self._prev_obs = obs.copy()
        return obs, reward, terminated, truncated, info


# ── Environment factory ───────────────────────────────────────────────────────

def make_env(env_id: str, reward_fn: callable, render: bool = False, seed: int = 0):
    """Returns a callable that creates one wrapped MuJoCo environment."""
    def _init():
        render_mode = "rgb_array" if render else None
        kwargs = dict(render_mode=render_mode, width=256, height=256)
        # Ant-v4 defaults to use_contact_forces=False in gymnasium 0.29 (27-dim obs).
        # Enable it to get 111-dim obs so contact-based reward terms and the
        # per-foot contact proxy obs[27:111].reshape(4,21)[:,0] work correctly.
        if "Ant" in env_id:
            kwargs["use_contact_forces"] = True
        env = gym.make(env_id, **kwargs)
        env = RewardWrapper(env, reward_fn)
        env.reset(seed=seed)
        return env
    return _init


# ── Main evaluation function ──────────────────────────────────────────────────

def run_candidate(
    reward_code: str,
    env_id: str,
    n_train_steps: int,
    n_eval_steps: int,
    render: bool = True,
    seed: int = 0,
    return_model: bool = False,
) -> dict:
    """
    Full training + evaluation cycle for one reward candidate.

    Returns dict with keys:
        reward_code, mean_reward, reward_curve, com_heights, contacts,
        forward_velocities, forward_velocity_proxy, frames, episode_length,
        component_log, success (bool), error (str or None),
        model (SB3 PPO instance, only if return_model=True)

    Never raises — all exceptions are caught and returned as success=False.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    # Forward velocity obs index: obs[4] for Humanoid, obs[5] for Ant
    fwd_vel_idx = 4 if "Humanoid" in env_id else 5

    result = {
        "reward_code": reward_code,
        "mean_reward": -999.0,
        "reward_curve": [],
        "com_heights": [],
        "contacts": [],
        "forward_velocities": [],
        "forward_velocity_proxy": 0.0,
        "frames": [],
        "episode_length": 0,
        "component_log": {},
        "success": False,
        "error": None,
        "model": None,
    }

    train_envs = None
    try:
        # Compile reward function
        local_ns = {}
        exec(reward_code, local_ns)
        if "compute_reward" not in local_ns:
            raise ValueError("Generated code does not define compute_reward()")
        reward_fn = local_ns["compute_reward"]

        # ── Training phase ────────────────────────────────────────────────────
        # Use SB3's SubprocVecEnv (multiprocessing) with DummyVecEnv fallback.
        # PPO MlpPolicy runs on CPU — GPU is reserved for CLIP + Ollama.
        # SubprocVecEnv uses fork on Linux so exec'd reward_fn is inherited
        # by child processes without pickling.
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        try:
            train_envs = SubprocVecEnv([
                make_env(env_id, reward_fn, render=False, seed=seed + i)
                for i in range(N_ENVS)
            ])
        except Exception:
            train_envs = DummyVecEnv([
                make_env(env_id, reward_fn, render=False, seed=seed + i)
                for i in range(N_ENVS)
            ])

        model = PPO(
            "MlpPolicy",
            train_envs,
            n_steps=512,
            batch_size=128,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            seed=seed,
            device="cpu",  # MlpPolicy is faster on CPU; keeps GPU free for CLIP+LLM
        )
        model.learn(total_timesteps=n_train_steps)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return result
    finally:
        if train_envs is not None:
            try:
                train_envs.close()
            except Exception:
                pass

    # ── Evaluation phase ──────────────────────────────────────────────────────
    eval_env = None
    try:
        eval_env_factory = make_env(env_id, reward_fn, render=render, seed=seed + 999)
        eval_env = eval_env_factory()

        rewards, com_heights, contacts, forward_velocities, frames = [], [], [], [], []
        component_log = {}
        step_count = 0

        obs, _ = eval_env.reset(seed=seed + 999)

        for step in range(n_eval_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

            rewards.append(float(reward))
            com_heights.append(float(obs[0]))           # obs[0] = z-height for both robots
            forward_velocities.append(float(obs[fwd_vel_idx]))  # actual forward velocity

            # Contact proxy: per-foot contact magnitude for Ant, generic last-8 for Humanoid
            if "Ant" in env_id and len(obs) >= 111:
                contact_val = float(np.sum(np.abs(obs[27:111].reshape(4, 21)[:, 0])))
            else:
                contact_val = float(np.sum(np.abs(obs[-8:])))
            contacts.append(contact_val)

            for k, v in info.get("reward_components", {}).items():
                if k not in component_log:
                    component_log[k] = []
                component_log[k].append(float(v) if not isinstance(v, str) else 0.0)

            if render and step % FRAME_SUBSAMPLE == 0:
                frame = eval_env.render()
                if frame is not None:
                    frames.append(frame)

            step_count += 1
            if terminated or truncated:
                break

        result.update({
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "reward_curve": rewards,
            "com_heights": com_heights,
            "contacts": contacts,
            "forward_velocities": forward_velocities,
            "forward_velocity_proxy": float(np.mean(forward_velocities)) if forward_velocities else 0.0,
            "frames": frames,
            "episode_length": step_count,
            "component_log": component_log,
            "success": True,
            "error": None,
            "model": model if return_model else None,
        })

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        if eval_env is not None:
            try:
                eval_env.close()
            except Exception:
                pass

    return result
