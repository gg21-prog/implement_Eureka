import re
import numpy as np
import requests

from src.config import HW, OLLAMA_URL, OLLAMA_TIMEOUT, LLM_TEMPERATURE, LLM_MAX_TOKENS


# ── Ollama HTTP client ────────────────────────────────────────────────────────

def call_ollama(prompt: str, temperature: float = None) -> str:
    """
    Calls local Ollama API. Returns raw text response.
    Raises ConnectionError if Ollama is not running.
    Raises requests.HTTPError on non-200 response.

    Does NOT extract code — that is the caller's job.
    """
    # Bug fix: `temperature or default` treats 0.0 as falsy.
    # Use explicit None check so temperature=0.0 (greedy) works correctly.
    temp = temperature if temperature is not None else LLM_TEMPERATURE

    payload = {
        "model": HW["llm_model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temp,
            "num_predict": LLM_MAX_TOKENS,
            "stop": ["```\n\n", "# End of function"],
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_URL}. "
            "Is Ollama running? Run: ollama serve"
        )
    data = resp.json()
    return data.get("response", "")


def extract_python_code(text: str) -> str:
    """
    Extracts Python code from LLM response.

    Strategy (tried in order):
    1. Content inside ```python ... ``` fence
    2. Content inside generic ``` ... ``` fence
    3. Everything from 'def compute_reward' to end of text
    4. Raw text fallback (LLM sometimes outputs bare code)
    """
    # Strategy 1: python-fenced
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 2: generic fence
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 3: bare function definition
    match = re.search(r"(def compute_reward.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 4: give it back as-is
    return text.strip()


# ── Observation layout strings ────────────────────────────────────────────────
# Injected verbatim into prompts so the LLM uses correct obs indices.

HUMANOID_OBS_LAYOUT = """
obs[0]       = torso z-position (height). Healthy: 0.7 < obs[0] < 2.0.
obs[1]       = torso forward tilt (0 = upright).
obs[2]       = torso side tilt (0 = upright).
obs[3]       = torso yaw rotation (0 = facing forward).
obs[4]       = forward velocity (x-axis) — PRIMARY LOCOMOTION SIGNAL
obs[5]       = lateral velocity (y-axis) — penalize abs(obs[5])
obs[6]       = vertical velocity (z-axis)
obs[7:10]    = angular velocities (roll, pitch, yaw) — penalize np.linalg.norm(obs[7:10])
obs[10:24]   = joint positions (14 joints)
obs[24:38]   = joint velocities (14 joints)
obs[38:84]   = actuator forces (avoid using in reward — noisy)
obs[84:376]  = contact/inertia features (avoid using in reward — shape-dependent)
action       = 17-dim, range [-0.4, 0.4]
Energy proxy = np.square(action).sum()  — range 0 to ~2.72
"""

ANT_OBS_LAYOUT = """
obs[0]       = torso z-position (height). Healthy: 0.2 < obs[0] < 1.0.
obs[1:5]     = torso orientation quaternion (qw, qx, qy, qz). obs[1] near 1.0 = upright.
obs[5]       = forward velocity (x-axis) — PRIMARY LOCOMOTION SIGNAL
obs[6]       = lateral velocity (y-axis) — penalize abs(obs[6])
obs[7]       = vertical velocity (z-axis)
obs[8:11]    = angular velocities — penalize np.linalg.norm(obs[8:11])
obs[11:19]   = joint positions (8 joints: 4 hips + 4 ankles)
obs[19:27]   = joint velocities (8 joints)
obs[27:111]  = contact forces (84 values). Per-foot: obs[27:111].reshape(4,21)[:,0]
action       = 8-dim, range [-1.0, 1.0]
Energy proxy = np.square(action).sum()  — range 0 to 8.0
"""

OBS_LAYOUTS = {
    "humanoid": HUMANOID_OBS_LAYOUT,
    "ant": ANT_OBS_LAYOUT,
}


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_generation_prompt(env_id: str, robot_type: str, obs_layout: str) -> str:
    return f"""You are an expert reinforcement learning reward engineer.
Your task is to write a Python reward function for a locomotion robot.

ENVIRONMENT: {env_id}
ROBOT TYPE: {robot_type}
TASK: Train the robot to walk or run forward with a stable, natural, energy-efficient gait.
The motion should look natural — symmetric limb movement, upright posture, consistent velocity.
Do NOT simply maximize forward velocity — the gait quality matters as much as speed.

OBSERVATION SPACE LAYOUT (use these exact indices):
{obs_layout}

REQUIRED FUNCTION SIGNATURE:
def compute_reward(obs, prev_obs, action, info, env) -> tuple[float, dict]:
    \"\"\"
    obs:      current observation (numpy array)
    prev_obs: previous observation (numpy array, same shape as obs)
    action:   action just applied (numpy array)
    info:     info dict from gymnasium step()
    env:      the gymnasium environment (do not call step() on it)
    Returns: (total_reward: float, components: dict[str, float])
    \"\"\"

REQUIREMENTS:
1. Return (total_reward, components_dict) — both are required
2. components_dict must map component name strings to individual float values
   Example: {{"forward_progress": 1.2, "posture_penalty": -0.3, "energy_penalty": -0.1}}
3. Use ONLY numpy operations on obs, prev_obs, action arrays
4. Do NOT import any modules inside the function
5. Do NOT call env.step() or env.reset()
6. Handle the case where prev_obs may equal obs (first step)
7. Include ALL of these reward components:
   - forward_progress: reward moving forward (positive)
   - posture_reward: reward upright stable torso (positive)
   - energy_penalty: penalize large actions/torques (negative)
   - lateral_penalty: penalize sideways movement (negative)
   - angular_penalty: penalize spinning/rotation (negative)
   - survival_bonus: small constant for staying alive (positive)
8. You MAY add additional components (gait symmetry, foot clearance, etc.)
9. Scale components so no single term dominates (total reward magnitude ~1-5 per step)

Write ONLY the Python function. No explanation, no imports outside the function.

```python
def compute_reward(obs, prev_obs, action, info, env) -> tuple:
"""


def build_reflection_prompt(
    reward_code: str,
    quant_stats: dict,
    visual_report: str,
    robot_type: str,
    obs_layout: str,
) -> str:
    component_lines = []
    for k, v in quant_stats.get("component_log", {}).items():
        if isinstance(v, list) and v:
            component_lines.append(
                f"  {k}: mean={np.mean(v):.4f}, std={np.std(v):.4f}, "
                f"min={np.min(v):.4f}, max={np.max(v):.4f}"
            )
    component_stats_str = "\n".join(component_lines) or "  (no component data)"

    return f"""You are an expert reinforcement learning reward engineer.
You previously wrote a reward function for {robot_type} locomotion.
It has been evaluated and both quantitative statistics and visual analysis are provided.
Your task is to REWRITE the reward function to fix the identified problems.

CURRENT REWARD FUNCTION:
```python
{reward_code}
```

OBSERVATION SPACE LAYOUT (same as before):
{obs_layout}

== QUANTITATIVE TRAINING STATISTICS ==
Mean reward per step:   {quant_stats.get('mean_reward', 0):.4f}
Episode length:         {quant_stats.get('episode_length', 0)} / {quant_stats.get('max_eval_steps', 500)} steps
  (short episode = early termination = robot fell or became unhealthy)
Mean CoM height:        {quant_stats.get('mean_com', 0):.4f}
  (Humanoid healthy: 0.7-2.0, Ant healthy: 0.2-1.0)
CoM height std:         {quant_stats.get('com_std', 0):.4f}
  (high std = bouncing = instability)
Forward velocity proxy: {quant_stats.get('forward_velocity_proxy', 0):.6f}
  (mean forward velocity from obs — obs[4] for Humanoid, obs[5] for Ant)

Reward component breakdown:
{component_stats_str}

{visual_report}

== DIAGNOSIS INSTRUCTIONS ==
Read the quantitative stats AND visual analysis together. Look for contradictions:
- High mean_reward but CONCERN flags in visual = reward hacking (robot games the reward)
- Short episode_length = robot falls early = add stronger survival/posture rewards
- Low mean CoM height = robot crouching = increase upright posture reward
- High CoM std = unstable bouncing = add smoothness/stability penalty
- POSSIBLE HACKING warnings = reduce the gamed component, add corrective term

== REWRITE REQUIREMENTS ==
1. Keep the same function signature: compute_reward(obs, prev_obs, action, info, env)
2. Return (total_reward, components_dict) — required
3. Fix the specific issues identified above
4. Do NOT remove existing components without replacing them with something better
5. If visual shows reward hacking in a component, reduce its weight and add a penalty
6. If episode is short, strengthen the survival/posture components
7. Add new components if the visual analysis identifies missing behaviors
8. Keep total reward magnitude in range ~1-10 per step

Write ONLY the rewritten Python function. No explanation.

```python
def compute_reward(obs, prev_obs, action, info, env) -> tuple:
"""
