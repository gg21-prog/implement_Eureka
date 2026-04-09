import os
import torch
import psutil

# ── Linux headless rendering ─────────────────────────────────────────────────
# Must be set before any MuJoCo env is created. EGL = GPU-accelerated headless.
# Override by setting MUJOCO_GL in your shell before running.
if os.environ.get("MUJOCO_GL") is None and os.name != "nt":
    os.environ["MUJOCO_GL"] = "egl"
    print("[config] Set MUJOCO_GL=egl (override with env var before import)")

# ── Eureka loop hyperparameters ──────────────────────────────────────────────
N_CANDIDATES     = 4        # reward candidates generated per iteration
N_ITERATIONS     = 5        # total evolutionary iterations
N_ENVS           = 8        # parallel MuJoCo environments per candidate
N_TRAIN_STEPS    = 50_000   # PPO timesteps per candidate evaluation
N_EVAL_STEPS     = 500      # evaluation rollout length (steps)
FRAME_SUBSAMPLE  = 4        # collect 1 frame every N eval steps
N_KEY_FRAMES     = 6        # frames passed to CLIP

# ── Validation ───────────────────────────────────────────────────────────────
DRY_RUN_STEPS    = 10       # steps for reward code dry-run validation
MAX_RETRY_SEEDS  = 1        # retry attempts on crash (1 = retry once)

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT   = 180      # seconds

# ── LLM generation params ────────────────────────────────────────────────────
LLM_TEMPERATURE  = 0.7
LLM_MAX_TOKENS   = 1500

# ── CLIP ─────────────────────────────────────────────────────────────────────
CLIP_MODEL_ID    = "openai/clip-vit-base-patch32"
CLIP_IMAGE_SIZE  = 256      # render resolution for MuJoCo (width=height)

# ── Output directories ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")


def detect_hardware() -> dict:
    """
    Detects available compute and selects appropriate LLM model.

    Selection logic:
      CUDA + VRAM >= 5.5 GB  →  llama3.1:8b,      device=cuda
      CUDA + VRAM <  5.5 GB  →  llama3.2:3b,      device=cuda
      MPS (Apple Silicon)    →  llama3.2:3b,      device=mps,  sequential=True
      CPU only               →  llama3.2:3b,      device=cpu,  sequential=True

    sequential_mode=True means: finish all simulation work, THEN run the LLM.
    On MPS this is mandatory (shared memory). On CUDA it's not needed because
    Ollama is a separate process with its own VRAM allocation.

    Note: total_memory / 1e9 gives decimal-GB. RTX 4060 8 GB reads as ~8.5,
    safely above the 5.5 threshold.
    """
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        device = "cuda"
        llm_model = "llama3.1:8b" if vram_gb >= 5.5 else "llama3.2:3b"
        sequential_mode = False
    elif torch.backends.mps.is_available():
        device = "mps"
        llm_model = "llama3.1:8b"
        sequential_mode = True  # shared memory — must finish sim before LLM
    else:
        device = "cpu"
        llm_model = "llama3.1:8b"
        sequential_mode = True

    total_ram_gb = psutil.virtual_memory().total / 1e9

    return {
        "device": device,
        "llm_model": llm_model,
        "clip_device": device,
        "sequential_mode": sequential_mode,
        "total_ram_gb": round(total_ram_gb, 1),
    }


HW = detect_hardware()

print(
    f"[config] device={HW['device']} | llm={HW['llm_model']} | "
    f"sequential={HW['sequential_mode']} | RAM={HW['total_ram_gb']}GB"
)
