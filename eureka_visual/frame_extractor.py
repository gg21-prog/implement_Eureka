import numpy as np
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
