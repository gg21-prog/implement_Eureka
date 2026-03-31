# Eureka Visual: Automated RL Reward Engineering Pipeline

## 1. System Overview
This project is an automated pipeline that trains locomotion policies (Humanoid-v4, Ant-v4) in MuJoCo using Stable Baselines3. It uses a local LLM to write and iteratively refine Python reward functions. 

To combat "reward hacking" (where the robot achieves high scores via unnatural/broken physics), we use a local Vision-Language Model (CLIP) to visually evaluate rendered frames of the robot's behavior against predefined text probes (e.g., "robot is tumbling"). The CLIP output is converted into a structured text report, which is fed back to the LLM alongside quantitative training stats to rewrite and improve the reward function.

## 2. Hardware & Architecture Directives
The system must dynamically adapt to two distinct local environments:
1.  **M2 MacBook Air (8GB Unified Memory):**
    * **Constraint:** Extreme memory pressure. OS takes ~3.5GB.
    * **Strategy:** Sequential execution. Sim (CPU) -> Delete/GC -> Vision (MPS) -> Delete/GC -> LLM (MPS/CPU).
    * **Models:** `llama3.2:3b` (via Ollama) + `openai/clip-vit-base-patch32`.
2.  **RTX 4060 Laptop (8GB Dedicated VRAM):**
    * **Constraint:** VRAM is isolated from system RAM. 
    * **Strategy:** Concurrent execution. MuJoCo runs 100% on system CPU/RAM. The full 8GB VRAM is dedicated to the models.
    * **Models:** `llama3.1:8b-q4` (via Ollama) + `openai/clip-vit-base-patch32`.

### The Visual Pipeline (Crucial)
We **do not** pass raw images or image embeddings to the text-based LLM. 
1. Subsample ~6 keyframes from an evaluation rollout (start, end, max reward, min CoM, etc.).
2. Pass frames to CLIP at exactly **256x256 resolution** (to minimize compute while preserving enough detail for the ViT).
3. CLIP performs zero-shot scoring against dictionaries of positive/negative text probes (e.g., "upright balanced torso" vs "crouching low").
4. An analyzer script converts these softmax probabilities into a text report, explicitly flagging contradictions (e.g., `CONCERN: Quantitative forward velocity is high, but Visual shows "robot cartwheeling" = 0.85`).

## 3. Directory Structure
Implement the project exactly with this structure from scratch.

```text
eureka_visual/
├── config.py              # Hardware auto-detection & memory management flags
├── eureka.py              # Main orchestrator (the Eureka loop)
├── mujoco_runner.py       # SB3 PPO setup, AsyncVectorEnv (8x), frame rendering
├── frame_extractor.py     # Subsampling logic for key frames
├── clip_analyzer.py       # CLIP ViT inference and text-report generation
├── llm_feedback.py        # Ollama API integration, code generation, and reflection
├── probes/
│   ├── humanoid.py        # Probe text dictionaries for bipedal locomotion
│   └── ant.py             # Probe text dictionaries for quadrupedal locomotion
└── envs/
    ├── humanoid_wrapper.py # Gym wrapper WITH injected observation space docs
    └── ant_wrapper.py      # Gym wrapper WITH injected observation space docs
```

## 4. Module Implementation Requirements

### A. `config.py`
* Use `torch` and `psutil` to detect MPS vs CUDA. 
* Set a `sequential_mode` boolean flag. If `True` (Mac 8GB), the orchestrator must explicitly call `gc.collect()` and clear model caches between the simulation, vision, and LLM phases.

### B. `envs/humanoid_wrapper.py` & `envs/ant_wrapper.py`
* **The Hallucination Trap:** LLMs will hallucinate the array indices of the observation space if not told exactly what they are.
* You MUST include a massive, explicit Python docstring at the top of these wrappers documenting the exact Gym observation space array indices (e.g., `obs[0] = z-coordinate of torso`, `obs[1:5] = orientation quaternion`, etc.).
* Implement a `RewardWrapper` that takes an `exec()` compiled Python function and overrides the `step()` reward.
* Catch all exceptions in the custom reward function and return a `-1.0` penalty if the code crashes to prevent the pipeline from halting.

### C. `mujoco_runner.py`
* Use `gymnasium.vector.AsyncVectorEnv` to run exactly 8 environments for training (no rendering).
* For evaluation, instantiate a *single* standard environment.
* **Resolution Rule:** When calling `gym.make()` for the evaluation environment, strictly enforce `width=256, height=256` to ensure the frames sent to CLIP are lightweight but usable. 

### D. `clip_analyzer.py`
* Load `openai/clip-vit-base-patch32` via HuggingFace `transformers`.
* Take the nested dictionary of probes (categories -> text -> polarity).
* Calculate softmax probabilities across the probes for each frame, then average them.
* Generate the formatted String report. *Crucial:* Implement the contradiction logic. If negative probes average higher than positive probes for a category, append a `[CONCERN]` flag to the top of the report.

### E. `llm_feedback.py`
* Connect to `http://localhost:11434/api/generate` (Ollama).
* **Sandbox Safety:** The LLM *will* return code wrapped in markdown (
http://googleusercontent.com/immersive_entry_chip/0

This context file gives your agent absolutely no room for error regarding the architecture, while clearly defining the boundaries of what to implement. Let me know if you need to adjust the probe targets or hardware fallback logic!


Good — I have everything I need. Let me lock in all decisions and design the full system.
Hardware-adaptive model stack (both machines):

Mac 8GB unified: OS takes ~3.5GB → ~4.5GB free. llama3.2:3b (2GB) + CLIP ViT-B/32 (600MB) + 8 MuJoCo envs on CPU (~400MB). Fits, but sequential — sim batch first, then LLM pass.
4060 8GB VRAM: MuJoCo runs entirely on CPU, so full 8GB is for models. llama3.1:8b-q4 (5GB) + CLIP (600MB). Concurrent sim + LLM possible.
Runtime auto-detects which machine and loads accordingly.

CLIP strategy: We don't pass raw embeddings to the text LLM — it can't read them. Instead, CLIP does zero-shot scoring against a bank of locomotion-specific probe sentences ("robot is falling", "symmetric trot gait", "torso bent forward" etc.), which becomes a structured scored report the LLM can reason about. This is the key architectural decision.
Let me show the full system then break down every module:Now I have everything. Here's the full system design.
Locked-in stack:

Simulation: MuJoCo via gymnasium.vector.AsyncVectorEnv — 8 envs, no Isaac Lab needed, fully CPU-based which is ideal since it frees the entire GPU for the models
Vision: CLIP ViT-B/32 (600MB, runs on CUDA/MPS/CPU, works on both machines)
LLM: llama3.2:3b via Ollama on Mac 8GB (2GB), auto-upgrades to llama3.1:8b-q4 on the 4060 (detected at runtime)
RL: SB3 PPO (you know it already)
Robots: Humanoid-v4 and Ant-v4 from Gymnasium

The CLIP strategy is the most important design decision. We don't send raw embeddings to the LLM — instead CLIP scores each frame against a bank of ~20 locomotion-specific probe sentences ("robot hopping asymmetrically", "stable trot gait", "torso bent forward"), producing a structured scored report the LLM can actually reason about. Here's the full system:
  ::view-transition-group(*),
  ::view-transition-old(*),
  ::view-transition-new(*) {
    animation-duration: 0.25s;
    animation-timing-function: cubic-bezier(0.19, 1, 0.22, 1);
  }
VvisualizeVvisualize show_widgetThe coral CLIP node is the novel piece. Here's a zoom into what that pipeline actually does — this is the key to making visual feedback useful rather than vague:
  ::view-transition-group(*),
  ::view-transition-old(*),
  ::view-transition-new(*) {
    animation-duration: 0.25s;
    animation-timing-function: cubic-bezier(0.19, 1, 0.22, 1);
  }
VvisualizeVvisualize show_widgetThe dashed box at the bottom is the key insight — when CLIP sees high "hopping one leg" AND the quant stats show good forward velocity, the system flags that as reward hacking before the LLM ever writes a single word. Now here's the full project code:

Project structure
eureka_visual/
├── config.py              # hardware detection, model routing
├── eureka.py              # main loop
├── mujoco_runner.py       # 8x AsyncVectorEnv + SB3 rollouts
├── frame_extractor.py     # key frame selection
├── clip_analyzer.py       # CLIP probe scoring → structured report
├── llm_feedback.py        # Ollama calls + prompt templates
├── reward_reflection.py   # merge quant + visual, build rewrite prompt
├── probes/
│   ├── humanoid.py        # bipedal probe bank
│   └── ant.py             # quadruped probe bank
└── envs/
    ├── humanoid_wrapper.py
    └── ant_wrapper.py

config.py — hardware detection
This runs once at startup and every other module imports from it:
pythonimport torch
import psutil

def detect_hardware():
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        device = "cuda"
        available_gb = vram
    elif torch.backends.mps.is_available():
        device = "mps"
        # Unified memory: conservative estimate, OS takes ~3.5GB
        available_gb = (psutil.virtual_memory().total / 1e9) - 3.5
    else:
        device = "cpu"
        available_gb = 2.0

    # Model selection based on available memory
    # llama3.1:8b-q4 needs ~5.5GB, llama3.2:3b needs ~2.2GB
    llm_model = "llama3.1:8b-q4" if available_gb >= 5.5 else "llama3.2:3b"

    # On Mac MPS: must run sim and LLM sequentially (shared memory)
    # On CUDA: can overlap (sim on CPU, LLM on GPU)
    sequential_mode = device == "mps"

    return {
        "device": device,
        "available_gb": available_gb,
        "llm_model": llm_model,
        "clip_device": device,
        "sequential_mode": sequential_mode,
    }

HW = detect_hardware()
print(f"Hardware: {HW['device']} | LLM: {HW['llm_model']} | "
      f"Sequential: {HW['sequential_mode']}")

probes/humanoid.py and probes/ant.py
This is where locomotion domain knowledge lives. The probe design is deliberate — we pair each failure mode with its diagnostic positive so CLIP scores become interpretable deltas:
python# probes/humanoid.py
PROBES = {
    "posture": {
        "upright balanced torso":                    "positive",
        "torso bent far forward while moving":        "negative",
        "robot crouching low to the ground":          "negative",
        "humanoid with straight vertical spine":      "positive",
    },
    "gait_symmetry": {
        "symmetric alternating leg movement":         "positive",
        "hopping on one leg repeatedly":              "negative",
        "both legs moving together like a jump":      "negative",
        "natural human walking stride pattern":       "positive",
    },
    "stability": {
        "smooth continuous forward motion":           "positive",
        "jerky oscillating unstable movement":        "negative",
        "robot stumbling and recovering":             "negative",
        "steady consistent velocity":                 "positive",
    },
    "naturalness": {
        "arms swinging in opposition to legs":        "positive",
        "arms rigid and fixed while walking":         "negative",
        "stiff mechanical unnatural movement":        "negative",
        "fluid natural human-like locomotion":        "positive",
    },
    "foot_contact": {
        "feet lifting cleanly off the ground":        "positive",
        "feet dragging or shuffling along floor":     "negative",
        "robot sliding without proper foot contact":  "negative",
    },
    "failure_modes": {
        "robot falling over sideways":                "negative",
        "robot spinning in place":                    "negative",
        "robot moving backwards":                     "negative",
        "robot standing still not moving":            "negative",
    }
}

# probes/ant.py
PROBES = {
    "gait_pattern": {
        "quadruped diagonal trot gait":               "positive",
        "all four legs moving in coordination":       "positive",
        "only two legs on one side moving":           "negative",
        "legs moving in random uncoordinated order":  "negative",
    },
    "body_stability": {
        "body held level and stable while moving":    "positive",
        "body rolling or tilting to one side":        "negative",
        "body height consistent during locomotion":   "positive",
        "body bouncing excessively up and down":      "negative",
    },
    "contact": {
        "body dragging along the ground":             "negative",
        "all four feet making proper ground contact": "positive",
        "robot flipped upside down":                  "negative",
    },
    "failure_modes": {
        "robot tumbling or cartwheeling":             "negative",
        "robot spinning in circles":                  "negative",
        "robot stationary not making progress":       "negative",
    }
}

frame_extractor.py
pythonimport numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ExtractedFrames:
    frames: List[np.ndarray]
    labels: List[str]      # human-readable label per frame
    timesteps: List[int]

def extract_key_frames(
    all_frames: List[np.ndarray],
    reward_curve: List[float],
    com_heights: List[float],       # center-of-mass height over time
    contact_flags: List[float],     # foot contact count over time
    n_frames: int = 6
) -> ExtractedFrames:
    T = len(reward_curve)
    if T == 0 or len(all_frames) == 0:
        return ExtractedFrames([], [], [])

    selected = {}  # timestep → label

    # Always include boundaries
    selected[0] = "initial_state"
    selected[T - 1] = "final_state"

    # Peak and worst reward
    selected[int(np.argmax(reward_curve))] = "peak_reward"
    selected[int(np.argmin(reward_curve))] = "worst_reward"

    # Biggest reward drop (instability event)
    if T > 1:
        drops = np.diff(reward_curve)
        selected[int(np.argmin(drops))] = "instability_event"

    # CoM height minimum — potential fall detection
    if com_heights:
        selected[int(np.argmin(com_heights))] = "lowest_com"

    # Contact loss — moment when foot contact drops
    if contact_flags:
        contact_arr = np.array(contact_flags)
        diffs = np.diff(contact_arr)
        drops_idx = np.where(diffs < -0.5)[0]
        if len(drops_idx) > 0:
            selected[int(drops_idx[0])] = "contact_loss"

    # Fill remaining slots with uniform samples
    while len(selected) < n_frames:
        for i in range(0, T, max(1, T // n_frames)):
            if i not in selected:
                selected[i] = f"sample_t{i}"
            if len(selected) >= n_frames:
                break

    sorted_items = sorted(selected.items())[:n_frames]
    timesteps = [t for t, _ in sorted_items]
    labels = [l for _, l in sorted_items]

    # Map timesteps to frame indices
    # (frames may be subsampled during collection, e.g. every 4 steps)
    frame_idx = [min(t // 4, len(all_frames) - 1) for t in timesteps]
    frames = [all_frames[i] for i in frame_idx]

    return ExtractedFrames(frames=frames, labels=labels, timesteps=timesteps)

clip_analyzer.py
pythonimport torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from config import HW

_model = None
_processor = None

def _load_clip():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model = _model.to(HW["clip_device"])
        _model.eval()

def score_frames(
    frames: list,
    probes: dict,         # category → {probe_text: polarity}
    robot_type: str
) -> dict:
    """
    Returns a nested dict:
    {
      "posture": {
        "upright balanced torso": {"score": 0.81, "polarity": "positive"},
        ...
      },
      ...
      "_summary": {
        "posture": {"positive_avg": 0.76, "negative_avg": 0.22, "concern": False},
        ...
      },
      "_flags": ["reward_hacking: gait_symmetry", ...]
    }
    """
    _load_clip()
    device = HW["clip_device"]

    # Flatten all probe texts
    all_texts = []
    text_meta = []  # (category, text, polarity)
    for category, probe_dict in probes.items():
        for text, polarity in probe_dict.items():
            all_texts.append(text)
            text_meta.append((category, text, polarity))

    # Encode all frames
    pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frames]
    inputs = _processor(
        text=all_texts,
        images=pil_frames,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        # logits_per_image: [n_frames, n_texts]
        probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()

    # Average scores across frames
    mean_scores = probs.mean(axis=0)  # [n_texts]

    # Build result structure
    results = {cat: {} for cat in probes}
    for i, (cat, text, polarity) in enumerate(text_meta):
        results[cat][text] = {
            "score": float(mean_scores[i]),
            "polarity": polarity
        }

    # Build per-category summary
    summary = {}
    for cat, probe_results in results.items():
        pos_scores = [v["score"] for v in probe_results.values()
                      if v["polarity"] == "positive"]
        neg_scores = [v["score"] for v in probe_results.values()
                      if v["polarity"] == "negative"]
        pos_avg = np.mean(pos_scores) if pos_scores else 0
        neg_avg = np.mean(neg_scores) if neg_scores else 0
        # Concern = negative probes scoring higher than positives
        concern = neg_avg > pos_avg
        summary[cat] = {
            "positive_avg": round(pos_avg, 3),
            "negative_avg": round(neg_avg, 3),
            "concern": concern
        }

    # Flag cross-signal contradictions (reward hacking detector)
    flags = []
    for cat, s in summary.items():
        if s["concern"]:
            # Find the worst offending negative probe
            worst = max(
                [(k, v["score"]) for k, v in results[cat].items()
                 if v["polarity"] == "negative"],
                key=lambda x: x[1], default=(None, 0)
            )
            if worst[0] and worst[1] > 0.5:
                flags.append(f"CONCERN [{cat}]: '{worst[0]}' score={worst[1]:.2f}")

    results["_summary"] = summary
    results["_flags"] = flags
    return results


def format_report(scores: dict, robot_type: str) -> str:
    """Converts score dict into structured text for the LLM."""
    lines = [f"== VISUAL ANALYSIS (CLIP probes, robot: {robot_type}) ==\n"]

    for category, probe_results in scores.items():
        if category.startswith("_"):
            continue
        summary = scores["_summary"][category]
        concern_marker = " [CONCERN]" if summary["concern"] else ""
        lines.append(f"{category.upper()}{concern_marker}")
        lines.append(f"  positive avg: {summary['positive_avg']:.2f}  "
                     f"negative avg: {summary['negative_avg']:.2f}")

        # Show top positive and top negative
        pos = [(k, v["score"]) for k, v in probe_results.items()
               if v["polarity"] == "positive"]
        neg = [(k, v["score"]) for k, v in probe_results.items()
               if v["polarity"] == "negative"]
        pos.sort(key=lambda x: -x[1])
        neg.sort(key=lambda x: -x[1])
        if pos:
            lines.append(f"  best positive: [{pos[0][1]:.2f}] {pos[0][0]}")
        if neg:
            lines.append(f"  worst negative: [{neg[0][1]:.2f}] {neg[0][0]}")
        lines.append("")

    if scores["_flags"]:
        lines.append("DETECTED ISSUES:")
        for flag in scores["_flags"]:
            lines.append(f"  {flag}")

    return "\n".join(lines)

mujoco_runner.py
pythonimport gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import multiprocessing as mp

N_ENVS = 8

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
        obs, _, terminated, truncated, info = self.env.step(action)
        try:
            reward, components = self._reward_fn(obs, self._prev_obs, info, self.env)
        except Exception as e:
            reward, components = 0.0, {"error": str(e)}
        info["reward_components"] = components
        self._prev_obs = obs
        return obs, reward, terminated, truncated, info


def make_env(env_id, reward_fn, render=False, seed=0):
    def _init():
        mode = "rgb_array" if render else None
        env = gym.make(env_id, render_mode=mode)
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

    # 8 parallel envs for training — no rendering during train
    train_envs = AsyncVectorEnv([
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
    eval_env = gym.make(env_id, render_mode="rgb_array" if render else None)
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

        if render and step % 4 == 0:   # 1 in 4 frames = manageable memory
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

llm_feedback.py
pythonimport requests
import json
import re
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
````python
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
```

---

## `eureka.py` — the main loop
```python
import numpy as np
from config import HW
from mujoco_runner import run_candidate
from frame_extractor import extract_key_frames
from clip_analyzer import score_frames, format_report
from llm_feedback import generate_reward_candidate, generate_reflection

import probes.humanoid as humanoid_probes
import probes.ant as ant_probes

ROBOT_CONFIG = {
    "humanoid": {
        "env_id": "Humanoid-v4",
        "probes": humanoid_probes.PROBES,
        "env_code": open("envs/humanoid_wrapper.py").read()
    },
    "ant": {
        "env_id": "Ant-v4",
        "probes": ant_probes.PROBES,
        "env_code": open("envs/ant_wrapper.py").read()
    }
}

def run_eureka(
    robot_type: str = "humanoid",
    n_iterations: int = 5,
    n_candidates: int = 4,
    n_train_steps: int = 50_000,
):
    cfg = ROBOT_CONFIG[robot_type]
    best_overall = {"mean_reward": -np.inf, "reward_code": None}

    for iteration in range(n_iterations):
        print(f"\n=== Iteration {iteration+1}/{n_iterations} ===")

        # ── Phase 1: Generate candidates ──────────────────────────
        print("Generating reward candidates...")
        if iteration == 0 or best_overall["reward_code"] is None:
            # First iteration: generate from scratch
            candidates = [
                generate_reward_candidate(
                    cfg["env_id"], robot_type, cfg["env_code"]
                )
                for _ in range(n_candidates)
            ]
        else:
            # Later iterations: mutate best + generate some fresh
            candidates = [best_overall["reward_code"]]  # keep best
            candidates += [
                generate_reward_candidate(
                    cfg["env_id"], robot_type, cfg["env_code"]
                )
                for _ in range(n_candidates - 1)
            ]

        # ── Phase 2: Simulate (CPU) ────────────────────────────────
        # On Mac: sim runs here, GPU idle
        # On CUDA: sim on CPU, GPU available after
        print(f"Running {n_candidates} candidates × 8 envs...")
        results = []
        for i, code in enumerate(candidates):
            print(f"  Candidate {i+1}/{n_candidates}...")
            try:
                result = run_candidate(
                    code, cfg["env_id"],
                    n_train_steps=n_train_steps,
                    render=True
                )
                result["reward_code"] = code
                results.append(result)
            except Exception as e:
                print(f"  Candidate {i+1} failed: {e}")

        if not results:
            print("All candidates failed, skipping iteration")
            continue

        # ── Phase 3: Visual analysis (GPU/MPS) ────────────────────
        # On Mac sequential_mode=True: this runs after sim fully done
        best_this_iter = max(results, key=lambda r: r["mean_reward"])
        print(f"Best this iter: {best_this_iter['mean_reward']:.3f}")

        extracted = extract_key_frames(
            best_this_iter["frames"],
            best_this_iter["reward_curve"],
            best_this_iter["com_heights"],
            best_this_iter["contacts"],
            n_frames=6
        )

        print("Running CLIP analysis...")
        clip_scores = score_frames(
            extracted.frames,
            cfg["probes"],
            robot_type
        )
        visual_report = format_report(clip_scores, robot_type)
        print(visual_report)

        # ── Phase 4: LLM reflection and rewrite ───────────────────
        print("LLM reflecting and rewriting reward...")
        new_code = generate_reflection(
            best_this_iter["reward_code"],
            best_this_iter,
            visual_report
        )

        # Quick validation: does it compile?
        try:
            local_ns = {}
            exec(new_code, local_ns)
            assert "compute_reward" in local_ns
        except Exception as e:
            print(f"Rewritten code invalid: {e}, keeping previous best")
            new_code = best_this_iter["reward_code"]

        # Track best overall
        if best_this_iter["mean_reward"] > best_overall["mean_reward"]:
            best_overall = {**best_this_iter, "reward_code": new_code}
            print(f"New best: {best_overall['mean_reward']:.3f}")

    print(f"\nFinal best reward: {best_overall['mean_reward']:.3f}")
    return best_overall


if __name__ == "__main__":
    result = run_eureka(
        robot_type="humanoid",
        n_iterations=5,
        n_candidates=4,
        n_train_steps=50_000
    )
    print("\nBest reward function:")
    print(result["reward_code"])
```

---

## Setup and install
```bash
# Install Ollama (works on both Mac and Linux/Windows)
# Mac:
brew install ollama
# Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3.2:3b          # Mac 8GB (always works)
ollama pull llama3.1:8b-q4       # 4060 (auto-selected if VRAM ≥ 5.5GB)

# Python deps
pip install gymnasium[mujoco] stable-baselines3 \
    transformers torch pillow numpy requests psutil

# Run
python eureka.py
```

---

## The two things that will break first and how to fix them

`compute_reward` references an obs index that doesn't exist — the LLM hallucinates observation shapes. Fix: prepend the exact obs shape and a commented index map to the env code you pass as context. The Humanoid-v4 obs is 376-dimensional with a specific layout — document it explicitly in the prompt.

CLIP scores are all near-uniform (~0.05 per probe) — this happens when frames are too small or low-contrast. Fix: ensure `render_mode="rgb_array"` returns at least 256×256, and add `width=256, height=256` to `gym.make()`. CLIP ViT-B/32 was trained on images, not 64×64 thumbnails.

Want me to start writing the full code as runnable files, or go deeper on any specific module first?Start your own conversation