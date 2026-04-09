from dataclasses import dataclass, field
from typing import List
import numpy as np

from src.config import FRAME_SUBSAMPLE


@dataclass
class ExtractedFrames:
    frames: List[np.ndarray]
    labels: List[str]
    timesteps: List[int]

    def is_empty(self) -> bool:
        return len(self.frames) == 0


def extract_key_frames(
    all_frames: List[np.ndarray],
    reward_curve: List[float],
    com_heights: List[float],
    contacts: List[float],
    n_frames: int = 6,
) -> ExtractedFrames:
    """
    Selects n_frames from a rollout at the most diagnostically useful timesteps.

    Selection priority (no duplicates — dict keyed by frame index):
      1. t=0                          initial state
      2. t=len-1                      final state
      3. argmax(reward_curve)         peak performance
      4. argmin(reward_curve)         worst moment
      5. argmin(diff(reward_curve))   sharpest reward drop (instability)
      6. argmin(com_heights)          lowest CoM height (fall/crouch)
      7. argmin(contacts)             minimum contact (aerial/slip)
      8. uniform samples              fill remaining slots

    Frame index mapping: frames are collected every FRAME_SUBSAMPLE steps,
    so rollout timestep t → frame index t // FRAME_SUBSAMPLE (clamped).
    """
    T = len(reward_curve)
    F = len(all_frames)

    if F == 0 or T == 0:
        return ExtractedFrames(frames=[], labels=[], timesteps=[])

    def t_to_f(t: int) -> int:
        return min(int(t) // FRAME_SUBSAMPLE, F - 1)

    selected = {}  # frame_index → label

    # Priorities 1-2: boundaries
    selected[0]     = "initial_state"
    selected[F - 1] = "final_state"

    # Priorities 3-4: reward extremes
    selected[t_to_f(int(np.argmax(reward_curve)))] = "peak_reward"
    selected[t_to_f(int(np.argmin(reward_curve)))] = "worst_reward"

    # Priority 5: sharpest reward drop
    if T > 1:
        diffs = np.diff(reward_curve)
        selected[t_to_f(int(np.argmin(diffs)))] = "instability_event"

    # Priority 6: lowest CoM height
    if com_heights:
        selected[t_to_f(int(np.argmin(com_heights)))] = "lowest_com"

    # Priority 7: minimum contact
    if contacts:
        selected[t_to_f(int(np.argmin(contacts)))] = "min_contact"

    # Priority 8: uniform fill — pick evenly spaced from unselected indices
    if len(selected) < n_frames:
        available = [fi for fi in range(F) if fi not in selected]
        needed = n_frames - len(selected)
        if available and needed > 0:
            # Evenly space across available indices
            indices = np.linspace(0, len(available) - 1, needed, dtype=int)
            for idx in indices:
                fi = available[idx]
                if fi not in selected:
                    selected[fi] = f"uniform_t{fi * FRAME_SUBSAMPLE}"

    # Sort by frame index, take first n_frames
    sorted_items = sorted(selected.items())[:n_frames]
    frame_indices = [fi for fi, _ in sorted_items]
    labels       = [label for _, label in sorted_items]
    frames       = [all_frames[fi] for fi in frame_indices]
    timesteps    = [fi * FRAME_SUBSAMPLE for fi in frame_indices]

    return ExtractedFrames(frames=frames, labels=labels, timesteps=timesteps)
