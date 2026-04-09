"""
Run this before your first Eureka run to verify the full setup.
Usage: MUJOCO_GL=egl python tests/test_setup.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    print("  All imports: OK")


def test_mujoco():
    import gymnasium as gym
    env = gym.make("Ant-v4", render_mode="rgb_array", width=256, height=256,
                   use_contact_forces=True)
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
    print("  MuJoCo: OK — Ant (111-dim) and Humanoid (376-dim) render at 256x256")


def test_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    import numpy as np
    from PIL import Image

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dummy = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    inputs = processor(text=["test"], images=[dummy], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    assert out.logits_per_image.shape == (1, 1)
    print("  CLIP: OK")


def test_ollama():
    import requests as req
    try:
        resp = req.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"  Ollama: OK — available models: {models}")
        if not any("llama" in m for m in models):
            print("  WARNING: No llama models found. Run: ollama pull llama3.1:8b-q4")
    except Exception as e:
        print(f"  Ollama: NOT available — {e}")
        print("  To start: ollama serve  |  To pull model: ollama pull llama3.1:8b-q4")


def test_tensorboard():
    from torch.utils.tensorboard import SummaryWriter
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        w = SummaryWriter(d)
        w.add_scalar("test/val", 1.0, 0)
        w.close()
    print("  TensorBoard: OK")


def test_hardware():
    from src.config import HW
    print(f"  Hardware: device={HW['device']} | llm={HW['llm_model']} | "
          f"RAM={HW['total_ram_gb']}GB | sequential={HW['sequential_mode']}")


if __name__ == "__main__":
    tests = [
        test_imports,
        test_hardware,
        test_mujoco,
        test_clip,
        test_ollama,
        test_tensorboard,
    ]
    failed = 0
    for t in tests:
        print(f"[{t.__name__}]")
        try:
            t()
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} checks passed.")
    sys.exit(0 if failed == 0 else 1)
