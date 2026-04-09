import json
import os
import numpy as np
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar and array types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
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
    Also saves best reward code to run_dir/best_reward_iter{iteration:02d}.py.

    Strips frames and numpy arrays from best_result before serialising —
    those are too large for JSON and can be reconstructed from the policy.

    Returns path to checkpoint file.
    """
    serializable = {
        k: v for k, v in best_result.items()
        if k not in ("frames", "model") and not isinstance(v, np.ndarray)
    }
    # Coerce any remaining numpy lists to plain Python floats
    for k in ("reward_curve", "com_heights", "contacts", "forward_velocities"):
        if k in serializable and isinstance(serializable[k], list):
            serializable[k] = [float(x) for x in serializable[k]]

    checkpoint = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "best_result": serializable,
        "all_iteration_summaries": all_iteration_summaries,
    }

    ckpt_path = os.path.join(run_dir, "checkpoint.json")
    with open(ckpt_path, "w") as f:
        json.dump(checkpoint, f, cls=NumpyEncoder, indent=2)

    # Save best reward code as a standalone .py for easy inspection
    code_path = os.path.join(run_dir, f"best_reward_iter{iteration:02d}.py")
    with open(code_path, "w") as f:
        f.write(f"# Iteration {iteration} — mean_reward={best_result.get('mean_reward', 0):.4f}\n")
        f.write(f"# Robot: {best_result.get('robot_type', 'unknown')}\n\n")
        f.write(best_result.get("reward_code", "# No code"))

    return ckpt_path


def load_checkpoint(run_dir: str) -> dict:
    """
    Loads checkpoint.json from run_dir.
    Returns the checkpoint dict, or None if no checkpoint exists.
    """
    ckpt_path = os.path.join(run_dir, "checkpoint.json")
    if not os.path.exists(ckpt_path):
        return None
    with open(ckpt_path, "r") as f:
        return json.load(f)
