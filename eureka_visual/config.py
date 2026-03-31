import torch
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
    llm_model = "llama3.1:8b-q4" if available_gb >= 5.5 else "gemma3:4b"

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
