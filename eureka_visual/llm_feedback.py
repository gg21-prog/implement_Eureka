import requests
import json
import re
import numpy as np
from config import HW

OLLAMA_URL = "http://localhost:11434/api/generate"

REWARD_GEN_PROMPT = """You are an expert reinforcement learning reward engineer.

Environment: {env_id}
Robot type: {robot_type}
Task: Stable natural gait — the robot should walk or run forward smoothly,
with natural posture, symmetric limb movement, and consistent velocity.
The motion should look natural and energy-efficient, not jerky or hacked.

Environment source code (available observations and actions):
{env_code}

Write a Python reward function for this task. Requirements:
- Function signature: def compute_reward(obs, prev_obs, info, env) -> tuple[float, dict]
- Return (total_reward, components_dict)
- components_dict maps component names to their individual float values
- Use only variables accessible from obs, prev_obs, info, and env
- Include components for: forward progress, posture stability, 
  energy efficiency, gait symmetry, survival bonus
- Do not use any imports inside the function

Return only the Python function, no explanation.
```python
"""

REFLECTION_PROMPT = """You are an expert RL reward engineer. Rewrite the reward function below.

Current reward function:
```python
{reward_code}
```

== QUANTITATIVE TRAINING STATS ==
Mean reward: {mean_reward:.3f}
Episode length: {episode_length} / {max_steps} steps
Mean CoM height: {mean_com:.3f}  (low = crouching or falling)
CoM height std: {com_std:.3f}    (high = bouncing/instability)
Forward velocity proxy: {velocity:.4f}

Reward component stats:
{component_stats}

== VISUAL ANALYSIS (CLIP probe scores) ==
{visual_report}

== INSTRUCTIONS ==
Identify where the reward function is failing based on BOTH the numbers AND
the visual analysis above. Pay special attention to CONCERN flags — these
indicate the robot's actual behavior contradicts what the reward values suggest.

Rewrite the reward function to fix these issues. Be specific:
- If visual shows asymmetry but numbers show good reward → add symmetry penalty
- If visual shows reward hacking → reduce the gamed component, add corrective terms
- If CoM is low → add upright posture reward
- If episode is short → reduce termination conditions or add survival bonus

Return only the Python function, no explanation.
```python
"""

def generate_reward_candidate(env_id, robot_type, env_code) -> str:
    prompt = REWARD_GEN_PROMPT.format(
        env_id=env_id,
        robot_type=robot_type,
        env_code=env_code
    )
    return _call_ollama(prompt)

def generate_reflection(reward_code, stats, visual_report) -> str:
    component_stats = "\n".join([
        f"  {k}: mean={np.mean(v):.3f}, std={np.std(v):.3f}"
        for k, v in stats.get("components", {}).items()
    ])
    prompt = REFLECTION_PROMPT.format(
        reward_code=reward_code,
        mean_reward=stats["mean_reward"],
        episode_length=stats["episode_length"],
        max_steps=500,
        mean_com=np.mean(stats["com_heights"]),
        com_std=np.std(stats["com_heights"]),
        velocity=stats["forward_velocity_proxy"],
        component_stats=component_stats or "  (not available)",
        visual_report=visual_report
    )
    return _call_ollama(prompt)

def _call_ollama(prompt: str) -> str:
    payload = {
        "model": HW["llm_model"],
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,  # CRITICAL for 8GB Mac: Unloads the model immediately after responding
        "options": {"temperature": 0.7, "num_predict": 1024}
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json()["response"]
    return _extract_code(raw)

def _extract_code(text: str) -> str:
    """Pulls Python code from LLM response."""
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: find def compute_reward
    match = re.search(r"(def compute_reward.*)", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
