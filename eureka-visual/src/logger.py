import json
import os
import time
import numpy as np
from datetime import datetime

from src.checkpointer import NumpyEncoder  # defined in checkpointer, not config

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
        logger.log_iteration(iteration, all_results, best_result, visual_report)
        logger.log_final(best_result)
        logger.close()
    """

    def __init__(self, run_dir: str, robot_type: str):
        self.run_dir = run_dir
        self.robot_type = robot_type
        self.start_time = time.time()
        self.iteration_log = []
        self.json_path = os.path.join(run_dir, "iteration_log.json")

        self.writer = None
        if TENSORBOARD_AVAILABLE:
            tb_dir = os.path.join(run_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(tb_dir)
            print(f"[logger] TensorBoard: tensorboard --logdir {tb_dir}")
        else:
            print("[logger] TensorBoard not available, skipping.")

    def log_iteration(
        self,
        iteration: int,
        all_results: list,
        best_result: dict,
        visual_report: str,
    ):
        elapsed = time.time() - self.start_time
        mean_r  = best_result.get("mean_reward", 0)
        ep_len  = best_result.get("episode_length", 0)
        com     = best_result.get("mean_com", 0)
        n_success = sum(1 for r in all_results if r.get("success", False))

        print(
            f"\n[iter {iteration+1}] best_reward={mean_r:.4f} | "
            f"ep_len={ep_len} | com_height={com:.3f} | "
            f"candidates={n_success}/{len(all_results)} ok | "
            f"elapsed={elapsed:.0f}s"
        )

        if self.writer:
            self.writer.add_scalar("best/mean_reward",        mean_r,    iteration)
            self.writer.add_scalar("best/episode_length",     ep_len,    iteration)
            self.writer.add_scalar("best/mean_com_height",    com,       iteration)
            self.writer.add_scalar("run/n_successful_candidates", n_success, iteration)

            for k, v in best_result.get("component_log", {}).items():
                if isinstance(v, list) and v:
                    self.writer.add_scalar(
                        f"components/{k}_mean", float(np.mean(v)), iteration
                    )

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
            "visual_report_summary": visual_report[:500],
        }
        self.iteration_log.append(record)
        with open(self.json_path, "w") as f:
            json.dump(self.iteration_log, f, cls=NumpyEncoder, indent=2)

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
