import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3 import PPO
import multiprocessing as mp

N_ENVS = 4

class RewardWrapper(gym.Wrapper):
    """Hot-swaps in a GPT-generated reward function."""
    def __init__(self, env, reward_fn):
        super().__init__(env)
        self._reward_fn = reward_fn
        self._prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        obs, reward_env, terminated, truncated, info = self.env.step(action)
        try:
            reward, components = self._reward_fn(obs, self._prev_obs, info, self.env)
        except Exception as e:
            reward, components = -1.0, {"error": str(e)}
        info["reward_components"] = components
        self._prev_obs = obs
        return obs, float(reward), terminated, truncated, info


def make_env(env_id, reward_fn, render=False, seed=0):
    def _init():
        mode = "rgb_array" if render else None
        env = gym.make(env_id, render_mode=mode, width=256, height=256) if render else gym.make(env_id)
        env = RewardWrapper(env, reward_fn)
        env.reset(seed=seed)
        return env
    return _init


def run_candidate(
    reward_code: str,
    env_id: str,
    n_train_steps: int = 50_000,
    n_eval_steps: int = 500,
    render: bool = True,
    seed: int = 0
) -> dict:
    """Trains one candidate reward, returns stats + frames."""

    # Compile reward function from LLM-generated code
    local_ns = {}
    exec(reward_code, local_ns)
    reward_fn = local_ns["compute_reward"]

    # 4 sequential envs for training — avoids multiprocessing memory bloat on 8GB RAM
    train_envs = DummyVecEnv([
        make_env(env_id, reward_fn, render=False, seed=seed + i)
        for i in range(N_ENVS)
    ])

    model = PPO(
        "MlpPolicy", train_envs,
        n_steps=512, batch_size=64,
        learning_rate=3e-4, verbose=0
    )
    model.learn(total_timesteps=n_train_steps)
    train_envs.close()

    # Single eval env with rendering for frame collection
    eval_env = gym.make(env_id, render_mode="rgb_array" if render else None, width=256, height=256) if render else gym.make(env_id)
    eval_env = RewardWrapper(eval_env, reward_fn)

    frames, rewards, com_heights, contacts = [], [], [], []
    obs, _ = eval_env.reset(seed=seed)

    for step in range(n_eval_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        rewards.append(float(reward))

        # Extract locomotion-specific signals from obs
        # Humanoid obs[0] = z (height), Ant obs[0] = z
        com_heights.append(float(obs[0]))

        # Contact info if available
        contacts.append(info.get("contact_count", 0))

        if render and step % 10 == 0:   # 1 in 10 frames = much lower memory footprint
            frames.append(eval_env.render())

        if done or truncated:
            break

    eval_env.close()

    # Compute component correlations with total reward
    components_log = {}  # populated by wrapper info during eval
    
    return {
        "reward_code": reward_code,
        "mean_reward": float(np.mean(rewards)),
        "reward_curve": rewards,
        "com_heights": com_heights,
        "contacts": contacts,
        "frames": frames,
        "episode_length": len(rewards),
        "forward_velocity_proxy": float(np.mean(np.diff(com_heights)) 
                                        if len(com_heights) > 1 else 0),
    }
