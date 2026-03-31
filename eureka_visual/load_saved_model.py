import gymnasium as gym
from stable_baselines3 import PPO
from mujoco_runner import RewardWrapper
import os

def main():
    if not os.path.exists("humanoid_walker.zip"):
        print("ERROR: humanoid_walker.zip not found! You must finish training via visualize_reward.py first.")
        return

    print("Loading the saved Brain (humanoid_walker.zip)...")
    model = PPO.load("humanoid_walker")

    print("Loading the Rulebook (best_reward.py)...")
    reward_code = open("best_reward.py").read()
    local_ns = {}
    exec(reward_code, local_ns)
    reward_fn = local_ns["compute_reward"]

    print("Opening MuJoCo...")
    env = gym.make("Humanoid-v4", render_mode="human")
    env = RewardWrapper(env, reward_fn)
    
    obs, _ = env.reset()
    try:
        while True:
            # The robot consults its loaded brain for the best exact movement!
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nClosed viewer.")
        pass
    env.close()

if __name__ == "__main__":
    main()
