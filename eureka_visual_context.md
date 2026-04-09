# CLAUDE CODE IMPLEMENTATION BRIEF
## Project: Eureka-Visual — LLM-Driven Reward Design with Visual Feedback for MuJoCo Locomotion

---

## HOW TO READ THIS DOCUMENT

This is a complete implementation brief for Claude Code. Every decision has already been made. Your job is to implement exactly what is described here — do not improvise architecture, do not substitute libraries, do not simplify modules "for now". Implement everything in the order specified in the IMPLEMENTATION ORDER section. When a module is complete, verify it against the acceptance criteria listed for that module before moving on.

If something is ambiguous, apply the most explicit interpretation described. If a section says "exactly like this", reproduce it exactly.

---

## 0. PREREQUISITES & SETUP

Before implementing anything, ensure your environment satisfies these requirements.

### Python version

Python 3.10 or higher is required. Check with:
```bash
python --version
```

### PyTorch CUDA install (MUST be done BEFORE pip install requirements.txt)

On Linux with CUDA (RTX 4060, RTX 3080, etc.), PyTorch must be installed with the correct CUDA index URL **before** installing the rest of requirements.txt. Installing the wrong torch binary will give CPU-only PyTorch even on a CUDA machine.

```bash
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:
```bash
pip install -r requirements.txt
```

On Mac (Apple Silicon), `pip install -r requirements.txt` is sufficient — MPS support is bundled in the standard torch wheel.

### Ollama install

```bash
# Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# Mac:
brew install ollama
```

Start the Ollama server in a separate terminal:
```bash
ollama serve
```

Pull the model appropriate for RTX 4060 8GB VRAM:
```bash
ollama pull llama3.1:8b-q4
```

For Mac 8GB or limited VRAM systems:
```bash
ollama pull llama3.2:3b
```

The system auto-detects your hardware and selects the right model at runtime.

### CRITICAL: Linux rendering (MUJOCO_GL)

**Before every run on Linux, you must set the `MUJOCO_GL` environment variable.** MuJoCo requires an OpenGL backend for `render_mode="rgb_array"`. Without this, rendering will fail with GL context errors.

```bash
export MUJOCO_GL=egl
```

- `egl` enables GPU-accelerated EGL rendering for MuJoCo on headless Linux (recommended, requires NVIDIA drivers).
- If EGL is not available (e.g., non-NVIDIA GPU or missing EGL libraries), fall back to software rendering:
  ```bash
  export MUJOCO_GL=osmesa
  ```
  `osmesa` is slower but works without a GPU. Install with: `sudo apt-get install libosmesa6-dev`.

Include this in all run commands:
```bash
MUJOCO_GL=egl python -m src.eureka --robot ant
MUJOCO_GL=egl python -m src.eureka --robot humanoid --iterations 5 --candidates 4
MUJOCO_GL=egl python -m src.eureka --robot humanoid --resume
```

You can also add `export MUJOCO_GL=egl` to your `~/.bashrc` or `~/.zshrc` to set it permanently.

**Note**: `src/config.py` will also attempt to set this automatically at import time (see Section 5), but explicitly setting it in the shell before running is more reliable because MuJoCo may be initialized by subprocess workers before `config.py` is imported.

### Verification

After setup, verify everything works:
```bash
MUJOCO_GL=egl python tests/test_setup.py
```

All 5 tests should pass.

---

## 1. PROJECT OVERVIEW

Build a system called **eureka-visual** that automatically discovers reward functions for MuJoCo locomotion tasks using a local LLM, then iteratively improves those reward functions using both quantitative training statistics and qualitative visual analysis via CLIP.

The system implements the Eureka algorithm (evolutionary reward search + LLM reflection) extended with a visual feedback branch: key frames from rollouts are scored against a bank of locomotion-specific probe sentences using CLIP ViT-B/32, producing a structured behavioral report that is merged with quantitative stats and fed back to the LLM for reward rewriting.

Everything runs locally. No cloud API calls anywhere in the system.

### What this system does per run

1. Takes a robot type (`humanoid` or `ant`) and task description as input
2. Generates N reward function candidates using a local LLM via Ollama
3. Trains each candidate using SB3 PPO across 8 parallel MuJoCo environments
4. Extracts key frames from the best candidate's evaluation rollout
5. Scores frames against locomotion-specific CLIP probe sentences
6. Merges CLIP visual report with quantitative training stats
7. Feeds merged feedback to LLM to rewrite/improve the reward function
8. Repeats steps 2-7 for N iterations (evolutionary loop)
9. Saves: best reward code, SB3 policy checkpoint, demo video, JSON logs, TensorBoard events

### Supported robots

- `Humanoid-v4` (MuJoCo, 376-dim observation, 17-dim action)
- `Ant-v4` (MuJoCo, 111-dim observation, 8-dim action)

---

## 2. LOCKED DECISIONS — DO NOT CHANGE THESE

| Decision | Value |
|---|---|
| Simulation | MuJoCo via `gymnasium`, `AsyncVectorEnv` 8 envs |
| RL algorithm | Stable Baselines3 PPO |
| Vision model | CLIP ViT-B/32 via HuggingFace `transformers` |
| Vision strategy | Probe scoring → structured text report (NOT raw embeddings) |
| LLM interface | Ollama HTTP API (`http://localhost:11434/api/generate`) |
| LLM model (Mac 8GB) | `llama3.2:3b` |
| LLM model (4060 8GB VRAM) | `llama3.1:8b-q4` |
| LLM model selection | Automatic at runtime via hardware detection |
| Cloud APIs | None. Zero. Everything local. |
| Probes/hyperparams | Hardcoded in Python |
| Config | YAML not used. CLI args + Python constants only. |
| On crash | Log error, retry once with different seed, then skip |
| Code validation | Compile check + 10-step dry-run before full training |
| Final outputs | Best reward `.py`, SB3 policy `.zip`, demo video `.mp4` |
| Logging | Terminal progress + TensorBoard + JSON per iteration + best reward code per iteration |
| Resume | `--resume` flag loads checkpoint JSON and continues |
| Directory layout | Structured: `src/`, `configs/`, `outputs/`, `tests/` |

---

## 3. DIRECTORY STRUCTURE

Create exactly this layout. Do not add, remove, or rename directories.

```
eureka-visual/
├── src/
│   ├── __init__.py
│   ├── config.py              # hardware detection, all constants
│   ├── eureka.py              # main loop, CLI entrypoint
│   ├── mujoco_runner.py       # env wrappers, parallel rollouts, SB3 training
│   ├── frame_extractor.py     # key frame selection from rollout data
│   ├── clip_analyzer.py       # CLIP probe scoring + report formatting
│   ├── llm_client.py          # Ollama HTTP client + prompt templates
│   ├── reward_validator.py    # compile check + 10-step dry-run
│   ├── checkpointer.py        # save/load run state for --resume
│   ├── logger.py              # terminal + TensorBoard + JSON logging
│   └── probes/
│       ├── __init__.py
│       ├── humanoid.py        # bipedal probe bank
│       └── ant.py             # quadruped probe bank
├── outputs/                   # all run artifacts land here (git-ignored)
│   └── .gitkeep
├── tests/
│   ├── test_setup.py          # verifies all deps + Ollama + MuJoCo work
│   ├── test_clip.py           # quick CLIP smoke test
│   ├── test_reward_validator.py
│   └── test_mujoco_runner.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. DEPENDENCY VERSIONS

Use these exact versions in `requirements.txt`. Do not use `>=` or unpinned versions.

> **WARNING (Linux + CUDA users)**: `torch==2.3.1` listed below is a placeholder. On Linux with CUDA you MUST install PyTorch first with the correct CUDA index URL (see Section 0) before running `pip install -r requirements.txt`. Failing to do this will install a CPU-only torch binary even on a CUDA machine.
>
> ```bash
> pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```

```
gymnasium[mujoco]==0.29.1
stable-baselines3==2.3.2
# torch==2.3.1  # NOTE: On Linux+CUDA install manually first — see Section 0
torch==2.3.1
transformers==4.44.2
Pillow==10.4.0
numpy==1.26.4
requests==2.32.3
psutil==6.0.0
tensorboard==2.17.0
imageio==2.35.1
imageio-ffmpeg==0.5.1
pyyaml==6.0.2
tqdm==4.66.5
```

**Platform notes:**
- On Mac (Apple Silicon), `torch==2.3.1` supports MPS natively. No extra install needed.
- On Linux with CUDA, user may need `torch==2.3.1+cu121` — note this in README.
- `imageio-ffmpeg` is required for `.mp4` rendering. It installs ffmpeg binaries automatically.
- `gymnasium[mujoco]` installs MuJoCo bindings. MuJoCo itself is downloaded automatically on first use.

---

## 5. HARDWARE DETECTION — `src/config.py`

This module runs once at import time and exposes a `HW` dict that every other module imports. It must be the first thing implemented because everything else depends on it.

### EGL rendering setup (add at top of file, after imports)

```python
import os
# Ensure EGL rendering on Linux headless environments
# Must be set before any MuJoCo environment is created
if os.environ.get("MUJOCO_GL") is None and os.name != "nt":
    os.environ["MUJOCO_GL"] = "egl"
    print("[config] Set MUJOCO_GL=egl (override with env var before import)")
```

### Constants defined in this file

```python
# ── Eureka loop hyperparameters ─────────────────────────────────────
N_CANDIDATES     = 4      # reward candidates generated per iteration
N_ITERATIONS     = 5      # total evolutionary iterations
N_ENVS           = 8      # parallel MuJoCo environments per candidate
N_TRAIN_STEPS    = 50_000 # PPO timesteps per candidate evaluation
N_EVAL_STEPS     = 500    # evaluation rollout length (steps)
FRAME_SUBSAMPLE  = 4      # collect 1 frame every N eval steps
N_KEY_FRAMES     = 6      # frames passed to CLIP

# ── Validation ──────────────────────────────────────────────────────
DRY_RUN_STEPS    = 10     # steps for reward code dry-run validation
MAX_RETRY_SEEDS  = 1      # retry attempts on crash (1 = retry once)

# ── Ollama ──────────────────────────────────────────────────────────
OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT   = 180    # seconds

# ── LLM generation params ───────────────────────────────────────────
LLM_TEMPERATURE  = 0.7
LLM_MAX_TOKENS   = 1500

# ── CLIP ────────────────────────────────────────────────────────────
CLIP_MODEL_ID    = "openai/clip-vit-base-patch32"
CLIP_IMAGE_SIZE  = 256    # render resolution for MuJoCo (width=height)

# ── Output directories ──────────────────────────────────────────────
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")
```

### Hardware detection function

```python
import torch
import psutil

def detect_hardware() -> dict:
    """
    Detects available compute and selects appropriate LLM model.
    Returns a dict consumed by all other modules.

    Selection logic:
    - CUDA available + VRAM >= 5.5GB → llama3.1:8b-q4, device=cuda
    - CUDA available + VRAM < 5.5GB  → llama3.2:3b, device=cuda
    - MPS available (Apple Silicon)  → llama3.2:3b, device=mps
      (8GB unified memory: OS takes ~3.5GB, ~4.5GB usable,
       llama3.2:3b=2.2GB + CLIP=0.6GB + sim=~0.4GB = ~3.2GB total, fits)
    - CPU only                        → llama3.2:3b, device=cpu

    sequential_mode=True on MPS because unified memory is shared between
    the LLM and simulation. On MPS: finish all sim work, then run LLM.
    On CUDA: LLM runs on GPU, sim runs on CPU — can overlap if needed.
    """
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        device = "cuda"
        llm_model = "llama3.1:8b-q4" if vram_gb >= 5.5 else "llama3.2:3b"
        sequential_mode = False
    elif torch.backends.mps.is_available():
        device = "mps"
        llm_model = "llama3.2:3b"
        sequential_mode = True   # CRITICAL: shared memory, must be sequential
    else:
        device = "cpu"
        llm_model = "llama3.2:3b"
        sequential_mode = True

    total_ram_gb = psutil.virtual_memory().total / 1e9

    hw = {
        "device": device,
        "llm_model": llm_model,
        "clip_device": device,
        "sequential_mode": sequential_mode,
        "total_ram_gb": round(total_ram_gb, 1),
    }
    return hw

HW = detect_hardware()
```

The `HW` dict must be importable as `from src.config import HW`. Print a one-line summary at import time:

```python
print(f"[config] device={HW['device']} | llm={HW['llm_model']} | "
      f"sequential={HW['sequential_mode']} | RAM={HW['total_ram_gb']}GB")
```

---

## 6. OBSERVATION SPACE REFERENCE

These are critical. LLM-generated reward code will hallucinate observation indices unless we explicitly pass this information in prompts. Every prompt template that asks the LLM to write reward code must include the relevant observation layout.

### Humanoid-v4 — 376-dimensional observation

```
obs[0]       = torso z-position (height above ground). Healthy range: 1.0–2.0.
               Falls below ~0.7. Reward upright = reward obs[0] close to 1.3.

obs[1]       = torso x-tilt (forward/back lean). 0 = upright.
obs[2]       = torso y-tilt (side lean). 0 = upright.
obs[3]       = torso z-rotation (yaw). 0 = facing forward.

obs[4:7]     = torso linear velocity (vx, vy, vz).
               obs[4] = forward velocity (x-axis). Maximize this for running.

obs[7:10]    = torso angular velocity (wx, wy, wz).
               High values = spinning/unstable.

obs[10:24]   = joint positions (14 joints: abdomen, hips, knees, ankles, shoulders, elbows).
obs[24:38]   = joint velocities (14 joints, same order).

obs[38:84]   = actuator forces (tendons, 46 values).

obs[84:376]  = cinert + cvel + qfrc_actuator + cfrc_ext (contact forces, body inertias).
               AVOID using these in reward functions — they are noisy and shape-dependent.
               Use obs[0:84] only for reward computation.

ACTION SPACE: 17-dimensional continuous, range [-0.4, 0.4].
              Corresponds to 17 joint actuators.
```

### Ant-v4 — 111-dimensional observation

```
obs[0]       = torso z-position (height). Healthy range: 0.2–1.0.
               Below ~0.2 = fallen. Reward obs[0] > 0.3.

obs[1:5]     = torso orientation quaternion (qw, qx, qy, qz).
               obs[1] close to 1.0 = upright. Penalize deviation.

obs[5:8]     = torso linear velocity (vx, vy, vz).
               obs[5] = forward velocity (x-axis). Primary locomotion signal.

obs[8:11]    = torso angular velocity (wx, wy, wz).
               High values = spinning. Penalize abs(obs[8:11]).sum().

obs[11:19]   = joint positions (8 joints: 4 hips, 4 ankles).
obs[19:27]   = joint velocities (8 joints, same order).

obs[27:111]  = contact forces (84 values, 4 feet × 21 contact features each).
               Use for contact detection: obs[27:111].reshape(4,21)[:,0] gives
               per-foot contact force magnitude. Positive = foot on ground.

ACTION SPACE: 8-dimensional continuous, range [-1.0, 1.0].
              Corresponds to 8 joint actuators (4 hips + 4 ankles).
```

### Key reward-relevant signals (use these in generated reward code)

| Signal | Humanoid | Ant |
|---|---|---|
| Forward velocity | `obs[4]` | `obs[5]` |
| Height | `obs[0]` | `obs[0]` |
| Side velocity (penalize) | `obs[5]` | `obs[6]` |
| Angular velocity (penalize) | `np.linalg.norm(obs[7:10])` | `np.linalg.norm(obs[8:11])` |
| Height healthy range | `0.7 < obs[0] < 2.0` | `0.2 < obs[0] < 1.0` |
| Energy (penalize) | `np.square(action).sum()` | `np.square(action).sum()` |

---

## 7. PROBE BANKS — `src/probes/humanoid.py` and `src/probes/ant.py`

These are the core of the visual analysis system. Each probe is a sentence describing a visual state. CLIP scores each frame against all probes and returns cosine similarities. The scoring architecture converts these scores into a structured behavioral report.

The probe design rule: every failure mode has a paired positive description. This enables the system to compute *relative* scores (positive vs negative within a category), which is far more meaningful than absolute CLIP similarity values.

### `src/probes/humanoid.py`

```python
"""
CLIP probe bank for Humanoid-v4 bipedal locomotion.

Structure: dict of category → dict of {probe_text: polarity}
polarity is "positive" (desired behavior) or "negative" (failure/undesired).

Design rationale:
- Categories map to distinct reward components so CLIP feedback can target
  specific reward terms by name.
- Probe sentences are phrased as visual descriptions, not goals.
  "robot standing upright" not "robot should stand upright".
- Each negative probe should have at least one positive counterpart in
  the same category so the summary score is interpretable as a delta.
"""

ROBOT_TYPE = "humanoid"
ENV_ID = "Humanoid-v4"

PROBES = {
    "posture": {
        "humanoid robot standing upright with vertical torso": "positive",
        "humanoid robot with straight spine walking forward":  "positive",
        "humanoid robot torso bent far forward":               "negative",
        "humanoid robot crouching low to the ground":          "negative",
        "humanoid robot leaning sharply to one side":          "negative",
    },
    "gait_symmetry": {
        "humanoid robot with symmetric alternating leg movement":   "positive",
        "humanoid robot with natural human walking stride":         "positive",
        "humanoid robot hopping repeatedly on one leg":             "negative",
        "humanoid robot with both legs moving simultaneously":      "negative",
        "humanoid robot dragging one leg while the other moves":    "negative",
    },
    "arm_coordination": {
        "humanoid robot arms swinging in opposition to legs":           "positive",
        "humanoid robot with natural arm swing during walking":         "positive",
        "humanoid robot arms completely rigid at sides while walking":  "negative",
        "humanoid robot arms flailing outward erratically":            "negative",
    },
    "foot_clearance": {
        "humanoid robot feet lifting cleanly off the ground each step":  "positive",
        "humanoid robot feet dragging along the floor surface":          "negative",
        "humanoid robot sliding feet without proper lift":               "negative",
    },
    "motion_smoothness": {
        "humanoid robot moving with smooth continuous forward motion":   "positive",
        "humanoid robot with steady consistent walking velocity":        "positive",
        "humanoid robot with jerky oscillating unstable movement":       "negative",
        "humanoid robot stumbling and recovering balance repeatedly":    "negative",
        "humanoid robot bouncing up and down excessively":              "negative",
    },
    "failure_modes": {
        "humanoid robot fallen on the ground":           "negative",
        "humanoid robot spinning in place":              "negative",
        "humanoid robot moving backward instead of forward": "negative",
        "humanoid robot standing completely still":      "negative",
        "humanoid robot tumbling sideways":              "negative",
    },
}

# Thresholds for flagging concerns in formatted report
CONCERN_THRESHOLD = 0.45   # negative probe score above this triggers a flag
HACKING_THRESHOLD = 0.12   # negative_avg - positive_avg delta triggers hacking warning
```

### `src/probes/ant.py`

```python
"""
CLIP probe bank for Ant-v4 quadruped locomotion.
Same structural conventions as humanoid.py.
"""

ROBOT_TYPE = "ant"
ENV_ID = "Ant-v4"

PROBES = {
    "gait_pattern": {
        "quadruped robot with diagonal trot gait pattern":              "positive",
        "quadruped robot with all four legs coordinated":               "positive",
        "quadruped robot with natural four-legged walking stride":      "positive",
        "quadruped robot with only two legs on same side moving":       "negative",
        "quadruped robot with legs moving in random uncoordinated way": "negative",
        "quadruped robot legs all moving at same time":                 "negative",
    },
    "body_stability": {
        "quadruped robot body held level and stable while moving":      "positive",
        "quadruped robot consistent body height during locomotion":     "positive",
        "quadruped robot body rolling or tilting sideways":             "negative",
        "quadruped robot body bouncing excessively up and down":        "negative",
        "quadruped robot body swinging side to side":                   "negative",
    },
    "foot_contact": {
        "quadruped robot with all four feet making proper ground contact": "positive",
        "quadruped robot with feet pressing firmly into the ground":       "positive",
        "quadruped robot body dragging along the ground surface":          "negative",
        "quadruped robot with feet slipping without traction":             "negative",
    },
    "forward_progress": {
        "quadruped robot moving steadily forward":                     "positive",
        "quadruped robot with consistent forward velocity":            "positive",
        "quadruped robot spinning in circles instead of going forward": "negative",
        "quadruped robot moving sideways instead of forward":          "negative",
        "quadruped robot stationary not making forward progress":      "negative",
    },
    "failure_modes": {
        "quadruped robot flipped upside down":                         "negative",
        "quadruped robot tumbling or cartwheeling":                    "negative",
        "quadruped robot fallen on its side":                          "negative",
        "quadruped robot collapsed on the ground":                     "negative",
    },
}

CONCERN_THRESHOLD = 0.45
HACKING_THRESHOLD = 0.12
```

---

## 8. MUJOCO RUNNER — `src/mujoco_runner.py`

This module handles everything related to simulation: environment creation, reward injection, parallel training, and evaluation rollout with frame collection.

### RewardWrapper class

```python
import gymnasium as gym
import numpy as np

class RewardWrapper(gym.Wrapper):
    """
    Wraps a MuJoCo env to use a dynamically injected reward function.

    The injected reward function signature must be:
        def compute_reward(obs, prev_obs, action, info, env) -> tuple[float, dict]
    Returns (total_reward, component_dict).

    On error in reward function: returns (0.0, {"_error": str(e)}).
    Does NOT suppress the error silently — logs it to component dict.

    prev_obs is None on the first step after reset.
    action is the action that was just applied (before obs was returned).
    """
    def __init__(self, env: gym.Env, reward_fn: callable):
        super().__init__(env)
        self._reward_fn = reward_fn
        self._prev_obs = None
        self._last_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs.copy()
        self._last_action = np.zeros(self.action_space.shape)
        return obs, info

    def step(self, action):
        self._last_action = action
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
```

### make_env factory

```python
def make_env(env_id: str, reward_fn: callable, render: bool = False, seed: int = 0):
    """Returns a callable that creates one wrapped MuJoCo environment."""
    def _init():
        render_mode = "rgb_array" if render else None
        env = gym.make(
            env_id,
            render_mode=render_mode,
            width=256,   # CRITICAL: must be 256 minimum for CLIP
            height=256,
        )
        env = RewardWrapper(env, reward_fn)
        env.reset(seed=seed)
        return env
    return _init
```

### run_candidate function

This is the core evaluation function. It trains a PPO policy with a given reward function, then runs an evaluation rollout to collect performance data and frames.

```python
from gymnasium.vector import AsyncVectorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import traceback

def run_candidate(
    reward_code: str,
    env_id: str,
    n_train_steps: int,
    n_eval_steps: int,
    render: bool = True,
    seed: int = 0,
) -> dict:
    """
    Full training + evaluation cycle for one reward candidate.

    Returns dict with keys:
        reward_code, mean_reward, reward_curve, com_heights, contacts,
        forward_velocities, frames, episode_length, forward_velocity_proxy,
        component_log, success (bool), error (str or None)

    forward_velocity_proxy is the mean of per-step forward velocity observations:
        obs[4] for Humanoid-v4 (vx on x-axis)
        obs[5] for Ant-v4 (vx on x-axis)
    The correct index is selected based on env_id.

    On exception: returns dict with success=False and error message.
    Does NOT raise.
    """
    result_template = {
        "reward_code": reward_code,
        "mean_reward": -999.0,
        "reward_curve": [],
        "com_heights": [],
        "contacts": [],
        "forward_velocities": [],
        "frames": [],
        "episode_length": 0,
        "forward_velocity_proxy": 0.0,
        "component_log": {},
        "success": False,
        "error": None,
    }

    try:
        # Compile reward function
        local_ns = {}
        exec(reward_code, local_ns)
        if "compute_reward" not in local_ns:
            raise ValueError("Generated code does not define compute_reward()")
        reward_fn = local_ns["compute_reward"]

        # ── Training phase: 8 parallel envs, no rendering ──────────
        try:
            train_envs = AsyncVectorEnv([
                make_env(env_id, reward_fn, render=False, seed=seed + i)
                for i in range(N_ENVS)
            ])
        except Exception:
            from gymnasium.vector import SyncVectorEnv
            train_envs = SyncVectorEnv([
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
        )
        try:
            model.learn(total_timesteps=n_train_steps)
        finally:
            train_envs.close()

        # ── Evaluation phase: single env with rendering ─────────────
        eval_env_factory = make_env(env_id, reward_fn, render=render, seed=seed + 999)
        eval_env = eval_env_factory()

        # Determine which obs index holds forward velocity for this env
        # Humanoid-v4: obs[4] = vx (forward velocity)
        # Ant-v4:      obs[5] = vx (forward velocity)
        fwd_vel_idx = 4 if "Humanoid" in env_id else 5

        frames, rewards, com_heights, contacts, forward_velocities, component_log = \
            [], [], [], [], [], {}

        obs, _ = eval_env.reset(seed=seed + 999)
        step_count = 0

        for step in range(n_eval_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

            rewards.append(float(reward))
            com_heights.append(float(obs[0]))  # obs[0] = z-height for both robots
            forward_velocities.append(float(obs[fwd_vel_idx]))

            # Contact signal: use last element of obs as proxy if not in info
            contact_val = info.get("contact_count", float(np.sum(np.abs(obs[-8:]))))
            contacts.append(float(contact_val))

            # Accumulate component log
            for k, v in info.get("reward_components", {}).items():
                if k not in component_log:
                    component_log[k] = []
                component_log[k].append(float(v) if not isinstance(v, str) else 0.0)

            # Collect frame every FRAME_SUBSAMPLE steps
            if render and step % FRAME_SUBSAMPLE == 0:
                frame = eval_env.render()
                if frame is not None:
                    frames.append(frame)

            step_count += 1
            if terminated or truncated:
                break

        eval_env.close()

        fwd_vel = float(np.mean(forward_velocities)) if forward_velocities else 0.0

        result_template.update({
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "reward_curve": rewards,
            "com_heights": com_heights,
            "contacts": contacts,
            "forward_velocities": forward_velocities,
            "frames": frames,
            "episode_length": step_count,
            "forward_velocity_proxy": fwd_vel,
            "component_log": component_log,
            "success": True,
            "error": None,
        })

    except Exception as e:
        result_template["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        result_template["success"] = False

    return result_template
```

### Important implementation notes for mujoco_runner.py

- Import `N_ENVS`, `N_TRAIN_STEPS`, `N_EVAL_STEPS`, `FRAME_SUBSAMPLE` from `src.config`
- `AsyncVectorEnv` requires the reward function to be picklable. The `exec`'d function IS picklable in Python 3.10+. If pickling fails, the code above falls back to `SyncVectorEnv` automatically:
  ```python
  try:
      train_envs = AsyncVectorEnv([make_env(...) for i in range(N_ENVS)])
  except Exception:
      from gymnasium.vector import SyncVectorEnv
      train_envs = SyncVectorEnv([make_env(...) for i in range(N_ENVS)])
  ```
- `train_envs.close()` must be called in a `finally` block to avoid zombie processes.
- The `width=256, height=256` arguments to `gym.make()` are CRITICAL for CLIP. Without them, MuJoCo renders at 64×64 by default and CLIP similarity scores become near-uniform noise.
- `model.predict(obs, deterministic=True)` — always deterministic during evaluation.

---

## 9. FRAME EXTRACTOR — `src/frame_extractor.py`

Selects the most informative frames from a rollout for CLIP analysis. The selection strategy ensures we always see: the initial state, the final state, the best and worst moments, instability events, and low-height events (potential falls).

### ExtractedFrames dataclass

```python
from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class ExtractedFrames:
    frames: List[np.ndarray]
    labels: List[str]
    timesteps: List[int]

    def is_empty(self) -> bool:
        return len(self.frames) == 0
```

### extract_key_frames function

```python
def extract_key_frames(
    all_frames: List[np.ndarray],
    reward_curve: List[float],
    com_heights: List[float],
    contacts: List[float],
    n_frames: int = 6,
) -> ExtractedFrames:
    """
    Selects n_frames from a rollout at the most diagnostically useful timesteps.

    Selection priority (in order, no duplicates):
    1. t=0                         — initial state
    2. t=len-1                     — final state
    3. argmax(reward_curve)        — peak performance
    4. argmin(reward_curve)        — worst moment
    5. argmin(diff(reward_curve))  — sharpest reward drop (instability)
    6. argmin(com_heights)         — lowest CoM height (fall/crouch)
    7. argmin(contacts)            — minimum contact (aerial/slip)
    8. uniform samples             — fill remaining slots

    Frame index mapping: frames are subsampled at FRAME_SUBSAMPLE rate
    during collection, so timestep t maps to frame index t // FRAME_SUBSAMPLE.
    """
    T = len(reward_curve)
    F = len(all_frames)

    if F == 0 or T == 0:
        return ExtractedFrames(frames=[], labels=[], timesteps=[])

    def t_to_f(t):
        """Map rollout timestep to frame index, clamped to valid range."""
        return min(int(t) // FRAME_SUBSAMPLE, F - 1)

    selected = {}  # frame_index → label (use dict to deduplicate)

    # Priority 1-2: boundaries
    selected[0]     = "initial_state"
    selected[F - 1] = "final_state"

    # Priority 3-4: reward extremes
    selected[t_to_f(np.argmax(reward_curve))] = "peak_reward"
    selected[t_to_f(np.argmin(reward_curve))] = "worst_reward"

    # Priority 5: sharpest reward drop
    if T > 1:
        diffs = np.diff(reward_curve)
        selected[t_to_f(int(np.argmin(diffs)))] = "instability_event"

    # Priority 6: lowest CoM
    if com_heights:
        selected[t_to_f(int(np.argmin(com_heights)))] = "lowest_com"

    # Priority 7: minimum contact
    if contacts:
        selected[t_to_f(int(np.argmin(contacts)))] = "min_contact"

    # Priority 8: uniform fill
    if len(selected) < n_frames:
        step = max(1, F // (n_frames - len(selected) + 1))
        for fi in range(0, F, step):
            if fi not in selected:
                selected[fi] = f"uniform_t{fi * FRAME_SUBSAMPLE}"
            if len(selected) >= n_frames:
                break

    # Sort by frame index, take first n_frames
    sorted_items = sorted(selected.items())[:n_frames]
    frame_indices = [fi for fi, _ in sorted_items]
    labels = [label for _, label in sorted_items]
    frames = [all_frames[fi] for fi in frame_indices]
    timesteps = [fi * FRAME_SUBSAMPLE for fi in frame_indices]

    return ExtractedFrames(frames=frames, labels=labels, timesteps=timesteps)
```

Import `FRAME_SUBSAMPLE` from `src.config`.

---

## 10. CLIP ANALYZER — `src/clip_analyzer.py`

The visual analysis module. Loads CLIP once, scores frames against probe sentences, and formats a structured text report that the LLM can reason about. This is the key architectural innovation — probe scores are interpreted categorically, not as raw numbers.

### Model loading (lazy singleton)

```python
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from src.config import HW, CLIP_MODEL_ID

_clip_model = None
_clip_processor = None

def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print(f"[clip] Loading {CLIP_MODEL_ID} on {HW['clip_device']}...")
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _clip_model = _clip_model.to(HW["clip_device"])
        _clip_model.eval()
        print(f"[clip] Model loaded.")
    return _clip_model, _clip_processor


def unload_clip():
    """Free CLIP from GPU memory before LLM inference. Call after score_frames()."""
    global _clip_model, _clip_processor
    _clip_model = None
    _clip_processor = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### score_frames function

```python
def score_frames(
    frames: list,
    probes: dict,
    robot_type: str,
) -> dict:
    """
    Scores frames against probe sentences using CLIP cosine similarity.

    Args:
        frames:     list of np.ndarray frames (H, W, 3) uint8
        probes:     probe bank dict from src/probes/{robot}.py
        robot_type: "humanoid" or "ant" (for report labeling)

    Returns a dict with structure:
    {
        "humanoid" or "ant": {
            "posture": {
                "upright balanced torso": {"score": 0.81, "polarity": "positive"},
                ...
            },
            ...
        },
        "_summary": {
            "posture": {
                "positive_avg": 0.76,
                "negative_avg": 0.22,
                "concern": False,
                "top_positive": ("upright balanced torso", 0.81),
                "top_negative": ("robot crouching", 0.33),
            },
            ...
        },
        "_flags": [
            "CONCERN [gait_symmetry]: 'hopping on one leg' score=0.71",
            ...
        ],
        "_hacking_warnings": [
            "POSSIBLE HACKING [gait_symmetry]: negative_avg 0.41 >> positive_avg 0.19",
        ]
    }

    Important implementation details:
    - All probe texts are encoded in a single batch (one CLIP forward pass per frame).
    - Scores are softmax probabilities (logits_per_image.softmax(dim=-1)).
    - Scores are averaged across all extracted frames.
    - Frames are converted to PIL before CLIP encoding.
    - Do NOT use logits directly — use softmax probabilities.
    """
    model, processor = _load_clip()
    device = HW["clip_device"]

    # Flatten all probe texts with metadata
    all_texts = []
    text_meta = []  # (category, text, polarity)
    for category, probe_dict in probes.items():
        for text, polarity in probe_dict.items():
            all_texts.append(text)
            text_meta.append((category, text, polarity))

    if not all_texts or not frames:
        return {"_summary": {}, "_flags": [], "_hacking_warnings": []}

    # Convert frames to PIL
    pil_frames = []
    for f in frames:
        if isinstance(f, np.ndarray):
            pil_frames.append(Image.fromarray(f.astype(np.uint8)))
        else:
            pil_frames.append(f)

    # Encode: one batch, all frames × all texts
    inputs = processor(
        text=all_texts,
        images=pil_frames,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image shape: [n_frames, n_texts]
        probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()

    # Average across frames
    mean_scores = probs.mean(axis=0)  # [n_texts]

    # Build results structure
    results = {robot_type: {}}
    for cat in probes:
        results[robot_type][cat] = {}

    for i, (cat, text, polarity) in enumerate(text_meta):
        results[robot_type][cat][text] = {
            "score": float(mean_scores[i]),
            "polarity": polarity,
        }

    # Build per-category summaries
    summary = {}
    for cat, probe_results in results[robot_type].items():
        pos_items = [(k, v["score"]) for k, v in probe_results.items()
                     if v["polarity"] == "positive"]
        neg_items = [(k, v["score"]) for k, v in probe_results.items()
                     if v["polarity"] == "negative"]
        pos_scores = [s for _, s in pos_items]
        neg_scores = [s for _, s in neg_items]
        pos_avg = float(np.mean(pos_scores)) if pos_scores else 0.0
        neg_avg = float(np.mean(neg_scores)) if neg_scores else 0.0
        concern = neg_avg > pos_avg

        top_pos = max(pos_items, key=lambda x: x[1]) if pos_items else ("none", 0.0)
        top_neg = max(neg_items, key=lambda x: x[1]) if neg_items else ("none", 0.0)

        summary[cat] = {
            "positive_avg": round(pos_avg, 4),
            "negative_avg": round(neg_avg, 4),
            "concern": concern,
            "top_positive": top_pos,
            "top_negative": top_neg,
        }

    # Build flags (specific high-scoring negative probes)
    from src.probes import humanoid as h_probes
    from src.probes import ant as a_probes
    probe_module = h_probes if robot_type == "humanoid" else a_probes
    concern_threshold = probe_module.CONCERN_THRESHOLD
    hacking_threshold = probe_module.HACKING_THRESHOLD

    flags = []
    hacking_warnings = []

    for cat, s in summary.items():
        top_neg_text, top_neg_score = s["top_negative"]
        if top_neg_score > concern_threshold:
            flags.append(
                f"CONCERN [{cat}]: '{top_neg_text}' score={top_neg_score:.2f}"
            )
        delta = s["negative_avg"] - s["positive_avg"]
        if delta > hacking_threshold:
            hacking_warnings.append(
                f"POSSIBLE HACKING [{cat}]: "
                f"negative_avg={s['negative_avg']:.2f} >> "
                f"positive_avg={s['positive_avg']:.2f} (delta={delta:.2f})"
            )

    results["_summary"] = summary
    results["_flags"] = flags
    results["_hacking_warnings"] = hacking_warnings
    return results
```

### format_report function

Converts the score dict to a structured text block for the LLM reflection prompt.

```python
def format_report(scores: dict, robot_type: str) -> str:
    """
    Converts CLIP score dict into structured text for LLM consumption.

    Output format is deliberately structured to help the LLM identify
    specific reward components to fix. Categories map to common reward
    component names so the LLM can make targeted modifications.
    """
    lines = [f"== VISUAL ANALYSIS REPORT (CLIP, robot: {robot_type}) ==\n"]

    robot_scores = scores.get(robot_type, {})
    summary = scores.get("_summary", {})

    for category, probe_results in robot_scores.items():
        s = summary.get(category, {})
        concern_str = " *** CONCERN ***" if s.get("concern", False) else ""
        lines.append(f"[{category.upper()}]{concern_str}")
        lines.append(
            f"  positive_avg={s.get('positive_avg', 0):.3f}  "
            f"negative_avg={s.get('negative_avg', 0):.3f}"
        )
        top_pos = s.get("top_positive", ("n/a", 0))
        top_neg = s.get("top_negative", ("n/a", 0))
        lines.append(f"  strongest positive: [{top_pos[1]:.3f}] {top_pos[0]}")
        lines.append(f"  strongest negative: [{top_neg[1]:.3f}] {top_neg[0]}")
        lines.append("")

    flags = scores.get("_flags", [])
    hacking = scores.get("_hacking_warnings", [])

    if flags or hacking:
        lines.append("DETECTED BEHAVIORAL ISSUES:")
        for f in flags:
            lines.append(f"  {f}")
        for h in hacking:
            lines.append(f"  {h}")
        lines.append("")

    if not flags and not hacking:
        lines.append("No critical behavioral issues detected.")

    return "\n".join(lines)
```

---

## 11. REWARD VALIDATOR — `src/reward_validator.py`

Every LLM-generated reward function is validated before full training. Two-stage: compile check, then a 10-step dry-run in a live environment.

```python
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
    Validates LLM-generated reward code in two stages:

    Stage 1 — Compile check:
      exec() the code. Verify compute_reward is defined.
      Verify it accepts correct signature via inspect.

    Stage 2 — Dry-run:
      Create one MuJoCo env (no render, no wrapper yet).
      Call compute_reward(obs, prev_obs, action, info, env) for 10 steps.
      Check that return value is (float, dict).
      Check that no exceptions are raised.

    Returns (is_valid: bool, message: str).
    message is empty string on success, error description on failure.

    This function never raises. All exceptions are caught and returned
    as failure messages.
    """
    # ── Stage 1: Compile ─────────────────────────────────────────────
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

    # ── Stage 2: Dry-run ─────────────────────────────────────────────
    env = None
    try:
        env = gym.make(env_id, render_mode=None)
        obs, info = env.reset(seed=seed)
        prev_obs = obs.copy()
        action = env.action_space.sample()

        for step in range(10):
            obs, _, terminated, truncated, info = env.step(action)

            try:
                result = reward_fn(obs, prev_obs, action, info, env)
            except Exception as e:
                return False, (
                    f"compute_reward raised exception at dry-run step {step}: "
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                )

            # Validate return type
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

            if np.isnan(reward_val) or np.isinf(reward_val):
                return False, (
                    f"compute_reward returned NaN or Inf at step {step}"
                )

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
```

---

## 12. LLM CLIENT — `src/llm_client.py`

All Ollama communication and prompt templates live here.

### Ollama call function

```python
import requests
import re
import json
from src.config import HW, OLLAMA_URL, OLLAMA_TIMEOUT, LLM_TEMPERATURE, LLM_MAX_TOKENS

def call_ollama(prompt: str, temperature: float = None) -> str:
    """
    Calls local Ollama API. Returns raw text response.
    Raises requests.exceptions.RequestException on network error.
    Raises ValueError if response is malformed.

    This function does NOT extract code — callers do that.
    """
    payload = {
        "model": HW["llm_model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature or LLM_TEMPERATURE,
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
            f"Is Ollama running? Run: ollama serve"
        )
    data = resp.json()
    return data.get("response", "")


def extract_python_code(text: str) -> str:
    """
    Extracts Python code from LLM response.

    Strategy (in order):
    1. Content between ```python and ``` fences
    2. Content between ``` and ``` fences (language-unspecified)
    3. Everything from 'def compute_reward' to end of text
    4. Raw text as fallback (LLM may sometimes output bare code)

    Strips leading/trailing whitespace from extracted code.
    """
    # Strategy 1: python-fenced block
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 2: generic fenced block
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 3: bare function
    match = re.search(r"(def compute_reward.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 4: raw fallback
    return text.strip()
```

### Prompt templates

These are the two main prompts. They are the most critical part of the system. Do not simplify, condense, or rephrase them. Include them verbatim.

#### Generation prompt (first iteration — from scratch)

```python
def build_generation_prompt(
    env_id: str,
    robot_type: str,
    obs_layout: str,
) -> str:
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
```

#### Reflection prompt (subsequent iterations — rewrite from feedback)

```python
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
            import numpy as np
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
```

### Observation layout strings (injected into prompts)

```python
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
```

---

## 13. CHECKPOINTER — `src/checkpointer.py`

Saves and loads run state so `--resume` can continue an interrupted run.

### Checkpoint schema

The checkpoint is a JSON file saved after every iteration. It contains everything needed to reconstruct run state: the best reward code, all iteration results (without frame arrays — too large), and the current iteration index.

```python
import json
import os
import numpy as np
from datetime import datetime

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_checkpoint(
    run_dir: str,
    iteration: int,
    best_result: dict,
    all_iteration_summaries: list,
) -> str:
    """
    Saves checkpoint to run_dir/checkpoint.json.
    Also saves best reward code to run_dir/best_reward_iter{iteration}.py.

    best_result must have keys: reward_code, mean_reward (minimum).
    frames are excluded from checkpoint (too large).
    Returns path to checkpoint file.
    """
    # Strip frames from best_result before serializing
    serializable_result = {
        k: v for k, v in best_result.items()
        if k != "frames" and not isinstance(v, np.ndarray)
    }
    # Ensure lists of numpy values are serializable
    for k in ["reward_curve", "com_heights", "contacts", "forward_velocities"]:
        if k in serializable_result and isinstance(serializable_result[k], list):
            serializable_result[k] = [float(x) for x in serializable_result[k]]

    checkpoint = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "best_result": serializable_result,
        "all_iteration_summaries": all_iteration_summaries,
    }

    ckpt_path = os.path.join(run_dir, "checkpoint.json")
    with open(ckpt_path, "w") as f:
        json.dump(checkpoint, f, cls=NumpyEncoder, indent=2)

    # Save best reward code as standalone .py
    code_path = os.path.join(run_dir, f"best_reward_iter{iteration:02d}.py")
    with open(code_path, "w") as f:
        f.write(f"# Iteration {iteration} — mean_reward={best_result.get('mean_reward', 0):.4f}\n")
        f.write(f"# Robot: {best_result.get('robot_type', 'unknown')}\n\n")
        f.write(best_result.get("reward_code", "# No code"))

    return ckpt_path


def load_checkpoint(run_dir: str) -> dict:
    """
    Loads checkpoint from run_dir/checkpoint.json.
    Returns checkpoint dict, or None if no checkpoint exists.
    """
    ckpt_path = os.path.join(run_dir, "checkpoint.json")
    if not os.path.exists(ckpt_path):
        return None
    with open(ckpt_path, "r") as f:
        return json.load(f)
```

---

## 14. LOGGER — `src/logger.py`

Handles all logging: terminal progress, TensorBoard events, and JSON iteration logs.

```python
import os
import json
import time
from datetime import datetime
from src.checkpointer import NumpyEncoder  # NumpyEncoder is defined in checkpointer.py

# Conditionally import TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class EurekaLogger:
    """
    Unified logger for a Eureka run.

    Usage:
        logger = EurekaLogger(run_dir, robot_type)
        logger.log_iteration(iteration, results, best_result, visual_report)
        logger.log_final(best_result)
        logger.close()
    """

    def __init__(self, run_dir: str, robot_type: str):
        self.run_dir = run_dir
        self.robot_type = robot_type
        self.start_time = time.time()
        self.iteration_log = []

        # TensorBoard
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            tb_dir = os.path.join(run_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(tb_dir)
            print(f"[logger] TensorBoard: tensorboard --logdir {tb_dir}")
        else:
            print("[logger] TensorBoard not available, skipping.")

        # JSON log file
        self.json_path = os.path.join(run_dir, "iteration_log.json")

    def log_iteration(
        self,
        iteration: int,
        all_results: list,
        best_result: dict,
        visual_report: str,
    ):
        """Logs one complete Eureka iteration."""
        elapsed = time.time() - self.start_time
        mean_r = best_result.get("mean_reward", 0)
        ep_len = best_result.get("episode_length", 0)
        com = best_result.get("mean_com", 0)

        # Terminal output
        n_success = sum(1 for r in all_results if r.get("success", False))
        print(f"\n[iter {iteration+1}] best_reward={mean_r:.4f} | "
              f"ep_len={ep_len} | com_height={com:.3f} | "
              f"candidates={n_success}/{len(all_results)} ok | "
              f"elapsed={elapsed:.0f}s")

        # TensorBoard
        if self.writer:
            self.writer.add_scalar("best/mean_reward", mean_r, iteration)
            self.writer.add_scalar("best/episode_length", ep_len, iteration)
            self.writer.add_scalar("best/mean_com_height", com, iteration)
            self.writer.add_scalar("run/n_successful_candidates", n_success, iteration)

            # Per-component scalars
            for k, v in best_result.get("component_log", {}).items():
                if isinstance(v, list) and v:
                    import numpy as np
                    self.writer.add_scalar(f"components/{k}_mean", float(np.mean(v)), iteration)

        # JSON record
        record = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "best_mean_reward": round(mean_r, 6),
            "best_episode_length": ep_len,
            "n_candidates_total": len(all_results),
            "n_candidates_success": n_success,
            "all_candidate_rewards": [
                round(r.get("mean_reward", -999), 4) for r in all_results
            ],
            "visual_report_summary": visual_report[:500],  # truncate for log size
        }
        self.iteration_log.append(record)
        with open(self.json_path, "w") as f:
            json.dump(self.iteration_log, f, indent=2)

    def log_final(self, best_result: dict):
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"RUN COMPLETE — {self.robot_type}")
        print(f"Best reward:    {best_result.get('mean_reward', 0):.4f}")
        print(f"Episode length: {best_result.get('episode_length', 0)}")
        print(f"Total time:     {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"Outputs in:     {self.run_dir}")
        print(f"{'='*60}")

        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()
```

---

## 15. VIDEO RENDERER

Video rendering is handled inline in `eureka.py` using `imageio`. Implement as a standalone function (not a class):

```python
def render_demo_video(
    frames: list,
    output_path: str,
    fps: int = 30,
) -> str:
    """
    Renders frames to MP4 video using imageio-ffmpeg.
    frames: list of np.ndarray (H, W, 3) uint8
    Returns output_path on success, raises on failure.
    """
    import imageio
    import numpy as np

    if not frames:
        raise ValueError("No frames to render")

    # Ensure all frames are uint8
    frames_uint8 = [f.astype(np.uint8) for f in frames]

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                 quality=8, macro_block_size=1)
    for frame in frames_uint8:
        writer.append_data(frame)
    writer.close()

    return output_path
```

---

## 16. MAIN LOOP — `src/eureka.py`

The orchestrator. This is the last file to implement.

### CLI interface

```
MUJOCO_GL=egl python -m src.eureka --robot humanoid --iterations 5 --candidates 4 --resume
```

Use `argparse`:

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Eureka Visual Reward Search")
    parser.add_argument(
        "--robot", type=str, choices=["humanoid", "ant"], default="humanoid",
        help="Robot type to train"
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help=f"Number of Eureka iterations (default: N_ITERATIONS from config)"
    )
    parser.add_argument(
        "--candidates", type=int, default=None,
        help=f"Reward candidates per iteration (default: N_CANDIDATES from config)"
    )
    parser.add_argument(
        "--train-steps", type=int, default=None,
        help="PPO training steps per candidate"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint in latest run directory"
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Existing run directory to resume from (required if --resume is set)"
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Run name (used for output directory). Auto-generated if not set."
    )
    return parser.parse_args()
```

### Run directory creation and resume helpers

```python
def create_run_dir(robot_type: str, run_name: str = None) -> str:
    """
    Creates outputs/{robot_type}_{timestamp}_{run_name}/ directory.
    Returns path to run directory.
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{robot_type}_{ts}"
    if run_name:
        name += f"_{run_name}"
    run_dir = os.path.join(OUTPUTS_DIR, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def find_latest_run_dir(robot_type: str) -> str:
    """
    Scans outputs/ for the most recently modified directory matching
    {robot_type}_* and returns its path.
    Returns None if no matching directory is found.
    """
    import os
    if not os.path.isdir(OUTPUTS_DIR):
        return None
    candidates = [
        os.path.join(OUTPUTS_DIR, d)
        for d in os.listdir(OUTPUTS_DIR)
        if d.startswith(f"{robot_type}_") and
           os.path.isdir(os.path.join(OUTPUTS_DIR, d))
    ]
    if not candidates:
        return None
    # Sort by modification time, most recent last
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]
```

When `--resume` is set:
- If `--run-dir` is also provided, use that path directly as the run directory.
- If `--run-dir` is not provided, call `find_latest_run_dir(robot_type)` to locate the most recent run directory for that robot.
- If no matching directory is found, raise an error with a clear message.

When `--resume` is not set, always create a new directory with `create_run_dir`.

Example wiring in `main()`:

```python
def main():
    args = parse_args()
    robot_type = args.robot
    n_iterations = args.iterations or N_ITERATIONS
    n_candidates = args.candidates or N_CANDIDATES
    n_train_steps = args.train_steps or N_TRAIN_STEPS

    if args.resume:
        if args.run_dir:
            run_dir = args.run_dir
        else:
            run_dir = find_latest_run_dir(robot_type)
            if run_dir is None:
                raise RuntimeError(
                    f"--resume specified but no existing run directory found for "
                    f"robot '{robot_type}' in {OUTPUTS_DIR}. "
                    f"Use --run-dir to specify an explicit path."
                )
        print(f"[resume] Resuming from: {run_dir}")
    else:
        run_dir = create_run_dir(robot_type, args.run_name)
        print(f"[run] Output directory: {run_dir}")

    run_eureka(
        robot_type=robot_type,
        n_iterations=n_iterations,
        n_candidates=n_candidates,
        n_train_steps=n_train_steps,
        resume=args.resume,
        run_dir=run_dir,
    )
```

### Robot config

```python
ROBOT_CONFIGS = {
    "humanoid": {
        "env_id": "Humanoid-v4",
        "obs_layout": OBS_LAYOUTS["humanoid"],
    },
    "ant": {
        "env_id": "Ant-v4",
        "obs_layout": OBS_LAYOUTS["ant"],
    },
}
```

### Main loop pseudocode (implement this exactly)

```
function run_eureka(robot_type, n_iterations, n_candidates, n_train_steps, resume, run_dir):

    cfg = ROBOT_CONFIGS[robot_type]
    probes = load_probes(robot_type)  # humanoid.PROBES or ant.PROBES

    best_overall = {mean_reward: -inf, reward_code: None, robot_type: robot_type}
    start_iteration = 0
    all_iteration_summaries = []

    if resume:
        ckpt = load_checkpoint(run_dir)
        if ckpt is not None:
            best_overall = ckpt["best_result"]
            start_iteration = ckpt["iteration"] + 1
            all_iteration_summaries = ckpt["all_iteration_summaries"]
            print(f"Resumed from iteration {start_iteration}")

    logger = EurekaLogger(run_dir, robot_type)

    for iteration in range(start_iteration, n_iterations):
        print(f"\n=== Eureka iteration {iteration+1}/{n_iterations} ===")

        # ── PHASE 1: Generate candidates ─────────────────────────────
        print(f"[iter {iteration+1}] Generating {n_candidates} reward candidates...")
        candidates = []

        for i in range(n_candidates):
            if iteration == 0 or best_overall["reward_code"] is None or i > 0:
                # Fresh generation on first iter or for additional candidates
                raw = call_ollama(build_generation_prompt(
                    cfg["env_id"], robot_type, cfg["obs_layout"]
                ))
            else:
                # First candidate of later iters: start from rewritten best
                raw = best_overall.get("rewritten_code", best_overall["reward_code"])

            code = extract_python_code(raw)
            candidates.append(code)

        # ── PHASE 2: Validate + train each candidate ─────────────────
        print(f"[iter {iteration+1}] Validating and training candidates...")
        results = []

        for i, code in enumerate(candidates):
            print(f"  Candidate {i+1}/{n_candidates}: validating...", end=" ", flush=True)

            is_valid, msg = validate_reward_code(code, cfg["env_id"], seed=iteration*100+i)
            if not is_valid:
                print(f"INVALID: {msg[:80]}")
                # Retry once with temperature=0.4 for more conservative output
                print(f"  Retrying candidate {i+1}...")
                raw2 = call_ollama(
                    build_generation_prompt(cfg["env_id"], robot_type, cfg["obs_layout"]),
                    temperature=0.4
                )
                code = extract_python_code(raw2)
                is_valid, msg = validate_reward_code(code, cfg["env_id"], seed=iteration*100+i+50)
                if not is_valid:
                    print(f"  Retry also invalid: {msg[:80]} — skipping.")
                    continue

            print(f"valid. Training {n_train_steps} steps...")

            result = run_candidate(
                reward_code=code,
                env_id=cfg["env_id"],
                n_train_steps=n_train_steps,
                n_eval_steps=N_EVAL_STEPS,
                render=True,
                seed=iteration * 1000 + i,
            )

            if not result["success"]:
                print(f"  Training failed: {result['error'][:120]}")
                # Retry once with different seed
                print(f"  Retrying with different seed...")
                result = run_candidate(
                    reward_code=code,
                    env_id=cfg["env_id"],
                    n_train_steps=n_train_steps,
                    n_eval_steps=N_EVAL_STEPS,
                    render=True,
                    seed=iteration * 1000 + i + 500,
                )
                if not result["success"]:
                    print(f"  Second attempt also failed — skipping.")
                    continue

            result["robot_type"] = robot_type
            results.append(result)
            print(f"  Candidate {i+1}: mean_reward={result['mean_reward']:.4f}, "
                  f"ep_len={result['episode_length']}")

        if not results:
            print(f"[iter {iteration+1}] All candidates failed. Skipping iteration.")
            continue

        # ── PHASE 3: Select best, extract frames ─────────────────────
        best_this_iter = max(results, key=lambda r: r["mean_reward"])
        best_this_iter["mean_com"] = float(np.mean(best_this_iter["com_heights"])) \
                                      if best_this_iter["com_heights"] else 0.0
        best_this_iter["com_std"] = float(np.std(best_this_iter["com_heights"])) \
                                     if best_this_iter["com_heights"] else 0.0
        best_this_iter["max_eval_steps"] = N_EVAL_STEPS

        extracted = extract_key_frames(
            best_this_iter["frames"],
            best_this_iter["reward_curve"],
            best_this_iter["com_heights"],
            best_this_iter["contacts"],
            n_frames=N_KEY_FRAMES,
        )

        # ── PHASE 4: CLIP analysis ────────────────────────────────────
        visual_report = "(no frames available for CLIP analysis)"
        if not extracted.is_empty():
            print(f"[iter {iteration+1}] Running CLIP analysis on {len(extracted.frames)} frames...")
            clip_scores = score_frames(extracted.frames, probes, robot_type)
            visual_report = format_report(clip_scores, robot_type)
            print(visual_report[:600])  # Print truncated report to terminal

        # ── PHASE 5: LLM reflection ───────────────────────────────────
        # Free CLIP VRAM before LLM call (8GB budget: CLIP=0.6GB, llama3.1:8b-q4=5GB)
        unload_clip()
        print(f"[iter {iteration+1}] LLM reflecting and rewriting reward...")
        raw_rewrite = call_ollama(build_reflection_prompt(
            best_this_iter["reward_code"],
            best_this_iter,
            visual_report,
            robot_type,
            cfg["obs_layout"],
        ))
        rewritten_code = extract_python_code(raw_rewrite)

        # Validate rewritten code
        is_valid, msg = validate_reward_code(rewritten_code, cfg["env_id"])
        if not is_valid:
            print(f"[iter {iteration+1}] Rewritten code invalid ({msg[:80]}). "
                  f"Keeping previous best.")
            rewritten_code = best_this_iter["reward_code"]

        best_this_iter["rewritten_code"] = rewritten_code

        # ── PHASE 6: Update best overall ─────────────────────────────
        if best_this_iter["mean_reward"] > best_overall.get("mean_reward", -np.inf):
            best_overall = {**best_this_iter}
            print(f"[iter {iteration+1}] New best overall: {best_overall['mean_reward']:.4f}")

        # ── PHASE 7: Log + checkpoint ─────────────────────────────────
        iter_summary = {
            "iteration": iteration,
            "best_mean_reward": best_this_iter["mean_reward"],
            "n_candidates": len(results),
        }
        all_iteration_summaries.append(iter_summary)

        logger.log_iteration(iteration, results, best_this_iter, visual_report)
        save_checkpoint(run_dir, iteration, best_overall, all_iteration_summaries)

    # ── FINAL OUTPUTS ─────────────────────────────────────────────────
    print("\n[final] Saving outputs...")

    # 1. Save final best reward code
    final_code_path = os.path.join(run_dir, "final_best_reward.py")
    with open(final_code_path, "w") as f:
        f.write(f"# Final best reward — {robot_type}\n")
        f.write(f"# mean_reward={best_overall.get('mean_reward', 0):.4f}\n\n")
        f.write(best_overall.get("reward_code", "# No code found"))

    # 2. Train final policy with best reward for more steps (2x training budget)
    print("[final] Training final policy with best reward...")
    final_result = run_candidate(
        reward_code=best_overall["reward_code"],
        env_id=cfg["env_id"],
        n_train_steps=n_train_steps * 2,
        n_eval_steps=1000,  # longer eval for video
        render=True,
        seed=99999,
    )

    # 3. Save SB3 policy — NOTE: SB3 model is not returned from run_candidate
    # To save the model, run_candidate must be modified to also return it.
    # Modify run_candidate to optionally return model:
    #   result["model"] = model  (before closing train_envs)
    # Then save here:
    if final_result.get("model") is not None:
        policy_path = os.path.join(run_dir, "final_policy.zip")
        final_result["model"].save(policy_path)
        print(f"[final] Policy saved: {policy_path}")

    # 4. Render demo video
    if final_result.get("frames"):
        video_path = os.path.join(run_dir, "demo.mp4")
        render_demo_video(final_result["frames"], video_path, fps=30)
        print(f"[final] Video saved: {video_path}")

    logger.log_final(best_overall)
    logger.close()

    return best_overall
```

### Modification to run_candidate for model saving

In `mujoco_runner.py`, after `model.learn(...)` and before `train_envs.close()`, add:

```python
result_template["model"] = model  # include model in return dict for final save
```

Only do this when the caller signals it wants the model (e.g., pass `return_model=False` as a parameter defaulting to False, and set True only for the final run).

---

## 17. IMPLEMENTATION ORDER

Implement files in exactly this order. Do not skip ahead. Each step has an acceptance criterion — verify it before proceeding.

### Step 1: `requirements.txt` and project scaffold
Create all directories and empty `__init__.py` files. Create `.gitignore` (ignore `outputs/`, `__pycache__/`, `*.pyc`, `*.zip`, `*.mp4`).

**Acceptance**: `find . -name "*.py" | head -20` shows the correct structure.

### Step 2: `src/config.py`
Implement hardware detection and all constants.

**Acceptance**: `python -c "from src.config import HW; print(HW)"` runs without error and prints a valid dict.

### Step 3: `src/probes/humanoid.py` and `src/probes/ant.py`
Copy probe banks exactly as specified. No changes.

**Acceptance**: `python -c "from src.probes.humanoid import PROBES; print(len(PROBES))"` prints 6.

### Step 4: `src/reward_validator.py`
Implement both validation stages.

**Acceptance**: `python tests/test_reward_validator.py` passes all cases (see test spec below).

### Step 5: `src/mujoco_runner.py`
Implement `RewardWrapper`, `make_env`, and `run_candidate`.

**Acceptance**: `MUJOCO_GL=egl python tests/test_mujoco_runner.py` runs one candidate for 1000 steps without hanging or crashing.

### Step 6: `src/frame_extractor.py`
Implement `ExtractedFrames` and `extract_key_frames`.

**Acceptance**: Function returns correct number of frames and no duplicate frame indices.

### Step 7: `src/clip_analyzer.py`
Implement model loading, `score_frames`, `format_report`, and `unload_clip`.

**Acceptance**: `python tests/test_clip.py` passes (see test spec).

### Step 8: `src/llm_client.py`
Implement `call_ollama`, `extract_python_code`, prompt builders, obs layout strings.

**Acceptance**: `call_ollama("Say hello")` returns a non-empty string (requires Ollama running).

### Step 9: `src/checkpointer.py` and `src/logger.py`
Implement both.

**Acceptance**: `save_checkpoint` creates a valid JSON file. `EurekaLogger` writes TensorBoard events.

### Step 10: `src/eureka.py`
Implement main loop and CLI.

**Acceptance**: `MUJOCO_GL=egl python -m src.eureka --robot ant --iterations 1 --candidates 1 --train-steps 5000` runs one full iteration end-to-end and produces output files.

### Step 11: `tests/` and `README.md`
Implement all test files and README.

---

## 18. TEST FILES

### `tests/test_setup.py`

Verifies everything is installed and accessible before a real run.

```python
"""
Run this before your first Eureka run to verify setup.
Usage: MUJOCO_GL=egl python tests/test_setup.py
"""
import sys

def test_imports():
    import gymnasium
    import stable_baselines3
    import torch
    import transformers
    import PIL
    import numpy
    import requests
    import psutil
    import imageio
    print("  All imports OK")

def test_mujoco():
    import gymnasium as gym
    env = gym.make("Ant-v4", render_mode="rgb_array", width=256, height=256)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (111,), f"Expected (111,), got {obs.shape}"
    frame = env.render()
    assert frame is not None and frame.shape == (256, 256, 3), \
        f"Expected (256, 256, 3), got {frame.shape if frame is not None else None}"
    env.close()

    env2 = gym.make("Humanoid-v4", render_mode="rgb_array", width=256, height=256)
    obs2, _ = env2.reset(seed=42)
    assert obs2.shape == (376,), f"Expected (376,), got {obs2.shape}"
    env2.close()
    print("  MuJoCo OK — Ant (111-dim) and Humanoid (376-dim)")

def test_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    import numpy as np
    from PIL import Image

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)
    pil = Image.fromarray(dummy_frame)
    inputs = processor(text=["test"], images=[pil], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    assert outputs.logits_per_image.shape == (1, 1)
    print("  CLIP OK")

def test_ollama():
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"  Ollama OK — available models: {models}")
        if not any("llama" in m for m in models):
            print("  WARNING: No llama models found. Run: ollama pull llama3.2:3b")
    except Exception as e:
        print(f"  Ollama NOT available: {e}")
        print("  To start Ollama: ollama serve")

def test_tensorboard():
    from torch.utils.tensorboard import SummaryWriter
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        w = SummaryWriter(d)
        w.add_scalar("test/val", 1.0, 0)
        w.close()
    print("  TensorBoard OK")

if __name__ == "__main__":
    tests = [test_imports, test_mujoco, test_clip, test_ollama, test_tensorboard]
    failed = 0
    for t in tests:
        print(f"Testing {t.__name__}...")
        try:
            t()
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    print(f"\n{len(tests)-failed}/{len(tests)} tests passed.")
    sys.exit(0 if failed == 0 else 1)
```

### `tests/test_reward_validator.py`

```python
"""Tests for reward code validation."""
import sys, os
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
    return 1.0  # missing dict
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
    for t in [test_valid, test_invalid_syntax, test_missing_function,
              test_wrong_return, test_nan_return]:
        try:
            t()
        except Exception as e:
            print(f"  {t.__name__}: FAIL — {e}")
```

### `tests/test_clip.py`

```python
"""Smoke test for CLIP probe scoring."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.clip_analyzer import score_frames, format_report
from src.probes.ant import PROBES

def test_clip_scores_ant():
    # Create 3 dummy frames (pure noise)
    frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
              for _ in range(3)]
    scores = score_frames(frames, PROBES, "ant")

    # Check structure
    assert "ant" in scores
    assert "_summary" in scores
    assert "_flags" in scores
    assert "_hacking_warnings" in scores

    # Check all categories present
    for cat in PROBES:
        assert cat in scores["ant"], f"Missing category: {cat}"
        assert cat in scores["_summary"], f"Missing summary for: {cat}"

    # Check scores are valid probabilities
    for cat, probe_results in scores["ant"].items():
        for text, data in probe_results.items():
            assert 0.0 <= data["score"] <= 1.0, \
                f"Score out of range: {data['score']} for '{text}'"

    # Check format_report produces non-empty string
    report = format_report(scores, "ant")
    assert len(report) > 100, "Report too short"
    assert "ANT" in report.upper()

    print("  test_clip_scores_ant: PASS")
    print(f"  Report preview:\n{report[:300]}")

if __name__ == "__main__":
    print("Testing CLIP analyzer (this will download CLIP on first run)...")
    test_clip_scores_ant()
```

### `tests/test_mujoco_runner.py`

```python
"""Smoke test for MuJoCo runner — runs one candidate for 1000 steps."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SIMPLE_REWARD = """
import numpy as np
def compute_reward(obs, prev_obs, action, info, env):
    forward_vel = obs[4]  # works for humanoid
    height = obs[0]
    energy = float(np.square(action).sum())
    posture = 1.0 if 0.7 < height < 2.0 else -1.0
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
    assert result["mean_reward"] != -999.0
    print(f"  test_run_candidate_smoke: PASS — mean_reward={result['mean_reward']:.4f}, "
          f"frames={len(result['frames'])}")

if __name__ == "__main__":
    print("Testing MuJoCo runner (requires MUJOCO_GL=egl on Linux)...")
    print("Note: uses Ant-v4 with 1000 train steps — takes ~1-2 min on GPU")
    test_run_candidate_smoke()
```

---

## 19. README.md

```markdown
# eureka-visual

Automatic reward function discovery for MuJoCo locomotion using LLM-driven
evolutionary search with CLIP-based visual behavioral feedback.

## What it does

Implements the Eureka algorithm with a visual extension:
1. A local LLM (via Ollama) generates candidate reward functions
2. SB3 PPO trains each candidate across 8 parallel MuJoCo environments
3. Key frames from the best rollout are scored against locomotion-specific
   CLIP probe sentences to produce a structured behavioral report
4. The LLM rewrites the reward function using both quantitative stats and
   the CLIP visual report
5. This loop repeats for N iterations, progressively improving the reward

Supported robots: `Humanoid-v4` (bipedal), `Ant-v4` (quadruped).
Everything runs locally — no cloud API calls.

## Setup

### 0. Prerequisites

Python 3.10+ required.

### 1. Install Python dependencies

**Linux + CUDA (RTX 4060, etc.) — install torch FIRST:**
```bash
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Mac (Apple Silicon):**
```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama

```bash
# Mac:
brew install ollama

# Linux:
curl -fsSL https://ollama.ai/install.sh | sh
```

Start the Ollama server:
```bash
ollama serve
```

Pull the required model (in a separate terminal):
```bash
# Mac 8GB RAM or limited GPU:
ollama pull llama3.2:3b

# RTX 4060 8GB VRAM or better:
ollama pull llama3.1:8b-q4
```

The system auto-detects your hardware and selects the right model.

### 3. Linux rendering setup (CRITICAL)

On Linux, MuJoCo requires an EGL context for GPU-accelerated rendering:
```bash
export MUJOCO_GL=egl
```

Add to `~/.bashrc` to set permanently. Without this, `render_mode="rgb_array"` fails.
If EGL is unavailable, use `export MUJOCO_GL=osmesa` (software rendering, slower).

### 4. Verify setup

```bash
MUJOCO_GL=egl python tests/test_setup.py
```

All 5 tests should pass. If Ollama isn't running, the Ollama test will warn
but not fail the setup — just make sure it's running before a real run.

## Usage

```bash
# Train humanoid (default: 5 iterations, 4 candidates, 50k steps each)
MUJOCO_GL=egl python -m src.eureka --robot humanoid

# Train ant with custom settings
MUJOCO_GL=egl python -m src.eureka --robot ant --iterations 3 --candidates 2 --train-steps 30000

# Resume interrupted run (auto-finds latest run directory for the robot)
MUJOCO_GL=egl python -m src.eureka --robot humanoid --resume

# Resume from a specific directory
MUJOCO_GL=egl python -m src.eureka --robot humanoid --resume --run-dir outputs/humanoid_20240101_120000
```

## Outputs

Each run creates `outputs/{robot}_{timestamp}/` containing:

| File | Description |
|---|---|
| `final_best_reward.py` | Best reward function found |
| `final_policy.zip` | SB3 PPO policy (loadable with `PPO.load()`) |
| `demo.mp4` | Video of final policy rollout |
| `best_reward_iter{N:02d}.py` | Best reward code after each iteration |
| `checkpoint.json` | Run state (for `--resume`) |
| `iteration_log.json` | Per-iteration stats |
| `tensorboard/` | TensorBoard event files |

View TensorBoard:
```bash
tensorboard --logdir outputs/
```

## Architecture

```
LLM (Ollama) ──generates──> reward.py
                                │
                         RewardWrapper
                                │
                    8× AsyncVectorEnv (MuJoCo)
                                │
                          SB3 PPO train
                                │
                       eval rollout + frames
                          /           \
               quant stats         CLIP probes
               (rewards,            (visual
               CoM, velocity)       behavior)
                          \           /
                        merged feedback
                                │
                      LLM reflection + rewrite
                                │
                           next iteration
```

## Hardware notes

| Machine | LLM model | Mode |
|---|---|---|
| RTX 4060 8GB | llama3.1:8b-q4 | concurrent (GPU=LLM, CPU=sim) |
| M2 MacBook Air 8GB | llama3.2:3b | sequential (shared memory) |
| M2 MacBook Air 16GB+ | llama3.2:3b | sequential |
| CPU only | llama3.2:3b | sequential (slow) |

On Mac with 8GB unified memory: the simulation and LLM share RAM.
The system automatically runs sim first, then LLM (sequential mode)
to avoid memory pressure. Expect ~20-40 min per iteration.

On RTX 4060 8GB: CLIP (~0.6GB) is unloaded from VRAM before the LLM
call (~5GB for llama3.1:8b-q4), keeping total VRAM usage within budget.
```

---

## 20. COMMON FAILURE MODES AND GUARDS

These are the most likely bugs. Implement guards for all of them.

### 1. CLIP gets 64×64 frames → uniform scores

**Symptom**: All CLIP probe scores are near-uniform (~0.05 each), no meaningful differentiation.
**Cause**: MuJoCo defaults to 64×64 renders if `width`/`height` not specified.
**Guard**: Always pass `width=256, height=256` to `gym.make()`. Add an assertion in `run_candidate`:
```python
if frames and frames[0].shape[:2] != (256, 256):
    raise ValueError(f"Frame shape is {frames[0].shape}, expected (256, 256, 3). "
                     f"Ensure gym.make() uses width=256, height=256.")
```

### 2. LLM generates reward code that references wrong obs indices

**Symptom**: `IndexError` or silently wrong reward values.
**Guard**: The obs layout strings in every prompt specify exact indices. The dry-run validator will catch index errors at step 0-9. No additional guard needed beyond what's specified.

### 3. `AsyncVectorEnv` hangs due to zombie worker processes

**Symptom**: Training hangs indefinitely after one candidate, no output.
**Cause**: `train_envs.close()` not called after training.
**Guard**: Wrap the training block in try/finally:
```python
try:
    model.learn(...)
finally:
    train_envs.close()
```

### 4. `exec()`'d reward function not picklable (Python < 3.10)

**Symptom**: `AttributeError: Can't pickle local object` when using `AsyncVectorEnv`.
**Cause**: `AsyncVectorEnv` spawns worker processes and needs to pickle the environment factory.
**Guard**: Test with `SyncVectorEnv` first if `AsyncVectorEnv` fails. Add a fallback:
```python
try:
    envs = AsyncVectorEnv([make_env(...) for _ in range(N_ENVS)])
except Exception:
    from gymnasium.vector import SyncVectorEnv
    envs = SyncVectorEnv([make_env(...) for _ in range(N_ENVS)])
```

### 5. LLM returns code with imports inside the function

**Symptom**: `ImportError` inside reward function at runtime.
**Cause**: LLM sometimes puts `import numpy as np` inside `compute_reward`.
**Guard**: The prompt explicitly says "Do NOT import any modules inside the function". The dry-run will catch this. Additionally, pre-inject `numpy` into the exec namespace:
```python
local_ns = {"np": __import__("numpy"), "numpy": __import__("numpy")}
exec(reward_code, local_ns)
```
This makes `import numpy as np` inside the function unnecessary and prevents import failures for the most common case.

### 6. Ollama timeout on slow hardware

**Symptom**: `requests.exceptions.Timeout` with `OLLAMA_TIMEOUT=180s`.
**Cause**: `llama3.1:8b-q4` on CPU can take >3min for a 1500-token response.
**Guard**: On `sequential_mode=True` (Mac/CPU), increase timeout to 300s. Auto-adjust in config:
```python
OLLAMA_TIMEOUT = 300 if HW["sequential_mode"] else 180
```

### 7. Video rendering fails on systems without ffmpeg

**Symptom**: `imageio` can't find ffmpeg encoder.
**Cause**: `imageio-ffmpeg` not installed or not in PATH.
**Guard**: Wrap video rendering in try/except and save raw frames as PNG fallback:
```python
try:
    render_demo_video(frames, video_path)
except Exception as e:
    print(f"[warn] Video render failed ({e}). Saving frames as PNG instead.")
    for i, frame in enumerate(frames[::10]):  # every 10th frame
        Image.fromarray(frame).save(os.path.join(run_dir, f"frame_{i:04d}.png"))
```

### 8. MuJoCo env returns None frame from render()

**Symptom**: `NoneType` error in frame extraction or CLIP.
**Cause**: `env.render()` returns None if `render_mode` was not set at `gym.make()` time.
**Guard**: Filter frames before passing to CLIP:
```python
frames = [f for f in raw_frames if f is not None and isinstance(f, np.ndarray)]
```

### 9. MuJoCo GL context error on Linux

**Symptom**: `mujoco.FatalError: gladLoadGL error` or `EGL` initialization failures.
**Cause**: `MUJOCO_GL` environment variable not set before MuJoCo is imported.
**Guard**: Set `MUJOCO_GL=egl` in shell before running (see Section 0). `src/config.py` also sets it at import time as a fallback, but shell-level export is more reliable for subprocess workers.

---

## 21. WHAT NOT TO DO

- Do NOT add `gymnasium.wrappers.RecordVideo` — it conflicts with manual frame collection.
- Do NOT use `gym.make("Humanoid-v4")` without `width=256, height=256` when rendering.
- Do NOT pass raw CLIP embedding vectors to the LLM — they are meaningless as text.
- Do NOT use `logits_per_image` directly — always apply `.softmax(dim=-1)` first.
- Do NOT call `env.step()` inside the reward function — only read `obs`, `info`, `env.unwrapped`.
- Do NOT skip the dry-run validation step to save time — it prevents wasted training runs.
- Do NOT use `SummaryWriter.add_text()` for the visual report — it floods TensorBoard. Use `add_scalar` only for numeric values.
- Do NOT run CLIP and the LLM simultaneously on Mac with 8GB RAM — sequential mode exists for this reason.
- Do NOT hardcode run directories — always use the timestamp-based `create_run_dir()` function.
- Do NOT modify probe sentences after the system is working — CLIP scores are relative within a probe bank; changing one sentence changes all scores.
- Do NOT compute `forward_velocity_proxy` as `np.diff(com_heights)` — that measures vertical z-height change, not forward movement. Use `obs[4]` (Humanoid) or `obs[5]` (Ant) directly.

---

*End of implementation brief.*
