import gymnasium as gym
from stable_baselines3 import PPO
from mujoco_runner import RewardWrapper
import os

def main():
    if not os.path.exists("best_reward.py"):
        print("Need a reward function to test!")
        return

    print("Opening MuJoCo viewer immediately (using an untrained random agent)...")
    reward_code = open("best_reward.py").read()
    local_ns = {}
    exec(reward_code, local_ns)
    reward_fn = local_ns["compute_reward"]

    env = gym.make("Humanoid-v4", render_mode="human")
    env = RewardWrapper(env, reward_fn)
    
    obs, _ = env.reset()
    try:
        while True:
            # We are taking random actions just to see the physics engine open!
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    env.close()

if __name__ == "__main__":
    main()
