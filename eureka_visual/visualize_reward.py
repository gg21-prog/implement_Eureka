import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import the wrapper and environment builder from your runner
from mujoco_runner import RewardWrapper, make_env

import os

def main():
    if not os.path.exists("best_reward.py"):
        print("ERROR: best_reward.py not found. Please run python3 eureka.py first!")
        return
        
    print("Loading LLM reward function from best_reward.py...")
    reward_code = open("best_reward.py").read()
    
    local_ns = {}
    exec(reward_code, local_ns)
    reward_fn = local_ns["compute_reward"]

    env_id = "Humanoid-v4"
    n_train_steps = 3_000_000  # Upped heavily! Millions are required for actual smooth bipedal coordination.
    n_envs = 4

    print(f"Training an agent on this reward for {n_train_steps} steps...")
    train_envs = DummyVecEnv([
        make_env(env_id, reward_fn, render=False, seed=i)
        for i in range(n_envs)
    ])

    model = PPO(
        "MlpPolicy", train_envs,
        n_steps=512, batch_size=64,  
        learning_rate=3e-4, verbose=1 # <--- Changed verbose to 1 so you can see it working!
    )
    model.learn(total_timesteps=n_train_steps)
    train_envs.close()

    print("Training complete! Saving the trained brain to 'humanoid_walker.zip'...")
    model.save("humanoid_walker")

    print("Opening MuJoCo graphical window to watch...")
    # render_mode="human" opens an actual native 3D window instead of rendering to internal arrays
    eval_env = gym.make(env_id, render_mode="human")
    eval_env = RewardWrapper(eval_env, reward_fn)
    
    obs, _ = eval_env.reset()
    try:
        for step in range(1500): # Watch it walk for 1500 frames
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            
            if done or truncated:
                obs, _ = eval_env.reset()
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    
    eval_env.close()

if __name__ == "__main__":
    main()
