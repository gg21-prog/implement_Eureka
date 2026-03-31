import numpy as np
import gc
import torch
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

def clean_memory():
    if HW["sequential_mode"]:
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_eureka(
    robot_type: str = "humanoid",
    n_iterations: int = 5,    # Increased from 3 back to 5 for better evolution!
    n_candidates: int = 4,    # Increased back up so the LLM tries multiple ideas at once
    n_train_steps: int = 150_000, # Increased so it tests rules more accurately against gravity
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

        clean_memory()

        # ── Phase 2: Simulate (CPU) ────────────────────────────────
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

        clean_memory()

        # ── Phase 3: Visual analysis (GPU/MPS) ────────────────────
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

        clean_memory()

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

        clean_memory()

    print(f"\nFinal best reward: {best_overall['mean_reward']:.3f}")
    return best_overall


if __name__ == "__main__":
    result = run_eureka(
        robot_type="humanoid",
        n_iterations=4,        # Match the defaults
        n_candidates=6,
        n_train_steps=150_000
    )
    print("\nBest reward function:")
    print(result["reward_code"])
    
    # Save the winner to a file automatically
    with open("best_reward.py", "w") as f:
        f.write(result["reward_code"])
    print("\nSaved best reward code to best_reward.py!")
