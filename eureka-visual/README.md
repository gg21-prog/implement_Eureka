# eureka-visual

LLM-driven reward function discovery for MuJoCo locomotion, extended with visual feedback via CLIP.

Implements the [Eureka](https://eureka-research.github.io/) algorithm with:
- **MuJoCo** (Gymnasium) instead of Isaac Gym — runs on a laptop
- **Ollama** (local LLM) instead of OpenAI API — fully open-source, no cost
- **CLIP visual feedback** — key frames from rollouts are scored against locomotion probe sentences and fed back to the LLM alongside quantitative stats

Supported robots: `Humanoid-v4`, `Ant-v4`

---

## Setup

### Requirements
- Python 3.10+
- NVIDIA GPU with CUDA 12.x (tested on RTX 4060 8GB)
- [Ollama](https://ollama.ai) installed and running

### 1. Install PyTorch with CUDA (must be done first)
```bash
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull the LLM model
```bash
ollama pull llama3.1:8b-q4   # for RTX 4060 8GB
# or for lower VRAM:
# ollama pull llama3.2:3b
```

### 4. Set MuJoCo rendering backend (Linux required)
```bash
export MUJOCO_GL=egl          # GPU-accelerated EGL (recommended)
# or: export MUJOCO_GL=osmesa  # software fallback
```
Add to `~/.bashrc` to make permanent.

### 5. Verify setup
```bash
MUJOCO_GL=egl python tests/test_setup.py
```

---

## Usage

```bash
# Train Ant with default settings (5 iterations, 4 candidates, 50k steps each)
MUJOCO_GL=egl python -m src.eureka --robot ant

# Train Humanoid with custom settings
MUJOCO_GL=egl python -m src.eureka --robot humanoid --iterations 3 --candidates 2 --train-steps 20000

# Quick smoke test (1 iteration, 1 candidate, 5000 steps)
MUJOCO_GL=egl python -m src.eureka --robot ant --iterations 1 --candidates 1 --train-steps 5000

# Resume a previous run (auto-detects latest directory)
MUJOCO_GL=egl python -m src.eureka --robot ant --resume

# Resume from a specific directory
MUJOCO_GL=egl python -m src.eureka --robot ant --resume --run-dir outputs/ant_20240101_120000
```

---

## Output structure

Each run creates `outputs/{robot}_{timestamp}/`:
```
outputs/ant_20240101_120000/
├── checkpoint.json              # resumable state after each iteration
├── iteration_log.json           # per-iteration metrics
├── best_reward_iter00.py        # best reward code after iteration 0
├── best_reward_iter01.py        # ... after iteration 1
├── final_best_reward.py         # best reward code across all iterations
├── final_policy.zip             # SB3 PPO policy trained with best reward
├── demo.mp4                     # evaluation rollout video
└── tensorboard/                 # TensorBoard event files
```

View training curves:
```bash
tensorboard --logdir outputs/
```

---

## How it works

1. **Generate** — Ollama LLM writes N reward function candidates given the observation layout
2. **Validate** — each candidate is compile-checked and dry-run for 10 steps before full training
3. **Train** — SB3 PPO trains each candidate for K steps across 8 parallel environments
4. **Extract** — key frames are selected from the best candidate's eval rollout (initial, final, peak reward, worst reward, instability event, lowest CoM)
5. **Analyse** — CLIP scores frames against locomotion probe sentences (posture, gait symmetry, foot contact, etc.) and produces a structured text report
6. **Reflect** — the LLM receives both quantitative stats and the CLIP visual report, and rewrites the reward function to fix identified problems
7. **Iterate** — repeat for N iterations; save best reward code and final policy

---

## Architecture notes

- PPO runs on **CPU** (`device="cpu"`) — MlpPolicy is faster on CPU and this frees the GPU for CLIP + Ollama
- CLIP is unloaded from GPU before the Ollama LLM call to avoid VRAM contention (CLIP ~0.6GB + llama3.1:8b-q4 ~5GB on 8GB card)
- `SubprocVecEnv` uses Linux fork semantics so exec'd reward functions are inherited without pickling
- `MUJOCO_GL=egl` is set automatically at import time if not already set (can override in shell)
