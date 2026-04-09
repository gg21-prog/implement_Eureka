"""
Eureka Visual — main loop and CLI entrypoint.

Usage:
    MUJOCO_GL=egl python -m src.eureka --robot ant --iterations 5 --candidates 4
    MUJOCO_GL=egl python -m src.eureka --robot humanoid --resume
    MUJOCO_GL=egl python -m src.eureka --robot ant --resume --run-dir outputs/ant_20240101_120000
"""
import argparse
import os
import numpy as np
import imageio

from src.config import (
    N_CANDIDATES, N_ITERATIONS, N_TRAIN_STEPS, N_EVAL_STEPS,
    N_KEY_FRAMES, OUTPUTS_DIR,
)
from src.mujoco_runner import run_candidate
from src.frame_extractor import extract_key_frames
from src.clip_analyzer import score_frames, format_report, unload_clip
from src.llm_client import (
    call_ollama, extract_python_code,
    build_generation_prompt, build_reflection_prompt,
    OBS_LAYOUTS,
)
from src.reward_validator import validate_reward_code
from src.checkpointer import save_checkpoint, load_checkpoint
from src.logger import EurekaLogger


# ── Robot configs ─────────────────────────────────────────────────────────────

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


# ── Directory helpers ─────────────────────────────────────────────────────────

def create_run_dir(robot_type: str, run_name: str = None) -> str:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{robot_type}_{ts}"
    if run_name:
        name += f"_{run_name}"
    run_dir = os.path.join(OUTPUTS_DIR, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def find_latest_run_dir(robot_type: str):
    """Returns the most recently modified outputs/{robot_type}_* directory, or None."""
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
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


# ── Video renderer ────────────────────────────────────────────────────────────

def render_demo_video(frames: list, output_path: str, fps: int = 30) -> str:
    """Renders frames to MP4. Returns output_path on success, raises on failure."""
    if not frames:
        raise ValueError("No frames to render")
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    for frame in frames:
        writer.append_data(frame.astype(np.uint8))
    writer.close()
    return output_path


# ── Main Eureka loop ──────────────────────────────────────────────────────────

def run_eureka(
    robot_type: str,
    n_iterations: int,
    n_candidates: int,
    n_train_steps: int,
    resume: bool,
    run_dir: str,
):
    cfg    = ROBOT_CONFIGS[robot_type]
    env_id = cfg["env_id"]
    obs_layout = cfg["obs_layout"]

    # Load probe bank
    if robot_type == "humanoid":
        from src.probes.humanoid import PROBES
    else:
        from src.probes.ant import PROBES

    best_overall = {
        "mean_reward": -np.inf,
        "reward_code": None,
        "robot_type": robot_type,
    }
    start_iteration = 0
    all_iteration_summaries = []

    if resume:
        ckpt = load_checkpoint(run_dir)
        if ckpt is not None:
            best_overall   = ckpt["best_result"]
            start_iteration = ckpt["iteration"] + 1
            all_iteration_summaries = ckpt["all_iteration_summaries"]
            print(f"[resume] Loaded checkpoint — starting at iteration {start_iteration + 1}")
        else:
            print("[resume] No checkpoint found — starting fresh")

    logger = EurekaLogger(run_dir, robot_type)

    for iteration in range(start_iteration, n_iterations):
        print(f"\n=== Eureka iteration {iteration+1}/{n_iterations} ===")

        # ── Phase 1: Generate candidates ──────────────────────────────────────
        print(f"[iter {iteration+1}] Generating {n_candidates} reward candidates...")
        candidates = []

        for i in range(n_candidates):
            # On first iteration, or for candidates beyond the first in later iters:
            # generate fresh. First candidate of later iterations starts from the
            # rewritten-best to provide a warm-start while others explore freely.
            use_rewrite = (
                iteration > 0
                and i == 0
                and best_overall.get("rewritten_code") is not None
            )
            if use_rewrite:
                raw = best_overall["rewritten_code"]
                code = extract_python_code(raw) if "```" in raw else raw
            else:
                raw = call_ollama(build_generation_prompt(env_id, robot_type, obs_layout))
                code = extract_python_code(raw)
            candidates.append(code)

        # ── Phase 2: Validate + train each candidate ───────────────────────────
        print(f"[iter {iteration+1}] Validating and training candidates...")
        results = []

        for i, code in enumerate(candidates):
            print(f"  Candidate {i+1}/{n_candidates}: validating...", end=" ", flush=True)

            is_valid, msg = validate_reward_code(code, env_id, seed=iteration * 100 + i)
            if not is_valid:
                print(f"INVALID: {msg[:80]}")
                print(f"  Retrying candidate {i+1} with lower temperature...")
                raw2 = call_ollama(
                    build_generation_prompt(env_id, robot_type, obs_layout),
                    temperature=0.4,
                )
                code = extract_python_code(raw2)
                is_valid, msg = validate_reward_code(
                    code, env_id, seed=iteration * 100 + i + 50
                )
                if not is_valid:
                    print(f"  Retry invalid: {msg[:80]} — skipping.")
                    continue

            print(f"valid. Training {n_train_steps} steps...")

            result = run_candidate(
                reward_code=code,
                env_id=env_id,
                n_train_steps=n_train_steps,
                n_eval_steps=N_EVAL_STEPS,
                render=True,
                seed=iteration * 1000 + i,
            )

            if not result["success"]:
                print(f"  Training failed: {result['error'][:120]}")
                print(f"  Retrying with different seed...")
                result = run_candidate(
                    reward_code=code,
                    env_id=env_id,
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
            print(
                f"  Candidate {i+1}: mean_reward={result['mean_reward']:.4f}, "
                f"ep_len={result['episode_length']}, "
                f"fwd_vel={result['forward_velocity_proxy']:.3f}"
            )

        if not results:
            print(f"[iter {iteration+1}] All candidates failed — skipping iteration.")
            continue

        # ── Phase 3: Select best, compute stats ───────────────────────────────
        best_this_iter = max(results, key=lambda r: r["mean_reward"])
        best_this_iter["mean_com"] = (
            float(np.mean(best_this_iter["com_heights"]))
            if best_this_iter["com_heights"] else 0.0
        )
        best_this_iter["com_std"] = (
            float(np.std(best_this_iter["com_heights"]))
            if best_this_iter["com_heights"] else 0.0
        )
        best_this_iter["max_eval_steps"] = N_EVAL_STEPS

        # ── Phase 4: Extract key frames ───────────────────────────────────────
        extracted = extract_key_frames(
            best_this_iter["frames"],
            best_this_iter["reward_curve"],
            best_this_iter["com_heights"],
            best_this_iter["contacts"],
            n_frames=N_KEY_FRAMES,
        )

        # ── Phase 5: CLIP visual analysis ─────────────────────────────────────
        visual_report = "(no frames available for CLIP analysis)"
        if not extracted.is_empty():
            print(f"[iter {iteration+1}] Running CLIP on {len(extracted.frames)} frames...")
            clip_scores = score_frames(extracted.frames, PROBES, robot_type)
            visual_report = format_report(clip_scores, robot_type)
            print(visual_report[:600])

        # Free CLIP VRAM before LLM call (8GB budget: CLIP~0.6GB + llama3.1:8b-q4~5GB)
        unload_clip()

        # ── Phase 6: LLM reflection + rewrite ────────────────────────────────
        print(f"[iter {iteration+1}] LLM reflecting and rewriting reward...")
        raw_rewrite = call_ollama(build_reflection_prompt(
            best_this_iter["reward_code"],
            best_this_iter,
            visual_report,
            robot_type,
            obs_layout,
        ))
        rewritten_code = extract_python_code(raw_rewrite)

        is_valid, msg = validate_reward_code(rewritten_code, env_id)
        if not is_valid:
            print(f"[iter {iteration+1}] Rewritten code invalid ({msg[:80]}) — keeping previous best.")
            rewritten_code = best_this_iter["reward_code"]

        best_this_iter["rewritten_code"] = rewritten_code

        # ── Phase 7: Update best overall ──────────────────────────────────────
        if best_this_iter["mean_reward"] > best_overall.get("mean_reward", -np.inf):
            best_overall = {**best_this_iter}
            print(f"[iter {iteration+1}] New best overall: {best_overall['mean_reward']:.4f}")

        # ── Phase 8: Log + checkpoint ──────────────────────────────────────────
        iter_summary = {
            "iteration": iteration,
            "best_mean_reward": best_this_iter["mean_reward"],
            "n_candidates": len(results),
        }
        all_iteration_summaries.append(iter_summary)
        logger.log_iteration(iteration, results, best_this_iter, visual_report)
        save_checkpoint(run_dir, iteration, best_overall, all_iteration_summaries)

    # ── Final outputs ─────────────────────────────────────────────────────────
    print("\n[final] Saving outputs...")

    final_code_path = os.path.join(run_dir, "final_best_reward.py")
    with open(final_code_path, "w") as f:
        f.write(f"# Final best reward — {robot_type}\n")
        f.write(f"# mean_reward={best_overall.get('mean_reward', 0):.4f}\n\n")
        f.write(best_overall.get("reward_code", "# No code found"))
    print(f"[final] Best reward code: {final_code_path}")

    # Train final policy with 2x budget and save model + video
    print("[final] Training final policy with best reward (2x steps)...")
    final_result = run_candidate(
        reward_code=best_overall["reward_code"],
        env_id=env_id,
        n_train_steps=n_train_steps * 2,
        n_eval_steps=1000,
        render=True,
        seed=99999,
        return_model=True,
    )

    if final_result.get("model") is not None:
        policy_path = os.path.join(run_dir, "final_policy.zip")
        final_result["model"].save(policy_path)
        print(f"[final] Policy saved: {policy_path}")

    if final_result.get("frames"):
        video_path = os.path.join(run_dir, "demo.mp4")
        try:
            render_demo_video(final_result["frames"], video_path, fps=30)
            print(f"[final] Video saved: {video_path}")
        except Exception as e:
            print(f"[final] Video render failed: {e}")

    logger.log_final(best_overall)
    logger.close()
    return best_overall


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Eureka Visual Reward Search")
    parser.add_argument(
        "--robot", type=str, choices=["humanoid", "ant"], default="humanoid",
        help="Robot type to train",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Number of Eureka iterations (default: N_ITERATIONS from config)",
    )
    parser.add_argument(
        "--candidates", type=int, default=None,
        help="Reward candidates per iteration (default: N_CANDIDATES from config)",
    )
    parser.add_argument(
        "--train-steps", type=int, default=None,
        help="PPO training steps per candidate (default: N_TRAIN_STEPS from config)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing run directory",
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Run directory to resume from (auto-detected from latest if not set)",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Suffix for output directory name (auto-generated if not set)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    robot_type    = args.robot
    n_iterations  = args.iterations  or N_ITERATIONS
    n_candidates  = args.candidates  or N_CANDIDATES
    n_train_steps = args.train_steps or N_TRAIN_STEPS

    if args.resume:
        run_dir = args.run_dir or find_latest_run_dir(robot_type)
        if run_dir is None:
            raise RuntimeError(
                f"--resume specified but no existing run directory found for "
                f"robot '{robot_type}' in {OUTPUTS_DIR}. "
                f"Provide --run-dir <path> to specify one explicitly."
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


if __name__ == "__main__":
    main()
