"""
CLIP probe bank for Humanoid-v4 bipedal locomotion.

Structure: dict of category -> dict of {probe_text: polarity}
polarity is "positive" (desired behavior) or "negative" (failure/undesired).

Design rationale:
- Categories map to distinct reward components so CLIP feedback can target
  specific reward terms by name.
- Probe sentences are phrased as visual descriptions, not goals.
  "robot standing upright" not "robot should stand upright".
- Each negative probe has at least one positive counterpart in the same
  category so the summary score is interpretable as a delta.
"""

ROBOT_TYPE = "humanoid"
ENV_ID = "Humanoid-v4"

PROBES = {
    "posture": {
        "humanoid robot standing upright with vertical torso": "positive",
        "humanoid robot with straight spine walking forward":  "positive",
        "humanoid robot torso bent far forward":               "negative",
        "humanoid robot crouching low to the ground":          "negative",
        "humanoid robot leaning sharply to one side":          "negative",
    },
    "gait_symmetry": {
        "humanoid robot with symmetric alternating leg movement":   "positive",
        "humanoid robot with natural human walking stride":         "positive",
        "humanoid robot hopping repeatedly on one leg":             "negative",
        "humanoid robot with both legs moving simultaneously":      "negative",
        "humanoid robot dragging one leg while the other moves":    "negative",
    },
    "arm_coordination": {
        "humanoid robot arms swinging in opposition to legs":           "positive",
        "humanoid robot with natural arm swing during walking":         "positive",
        "humanoid robot arms completely rigid at sides while walking":  "negative",
        "humanoid robot arms flailing outward erratically":            "negative",
    },
    "foot_clearance": {
        "humanoid robot feet lifting cleanly off the ground each step":  "positive",
        "humanoid robot feet dragging along the floor surface":          "negative",
        "humanoid robot sliding feet without proper lift":               "negative",
    },
    "motion_smoothness": {
        "humanoid robot moving with smooth continuous forward motion":   "positive",
        "humanoid robot with steady consistent walking velocity":        "positive",
        "humanoid robot with jerky oscillating unstable movement":       "negative",
        "humanoid robot stumbling and recovering balance repeatedly":    "negative",
        "humanoid robot bouncing up and down excessively":              "negative",
    },
    "failure_modes": {
        "humanoid robot fallen on the ground":                "negative",
        "humanoid robot spinning in place":                   "negative",
        "humanoid robot moving backward instead of forward":  "negative",
        "humanoid robot standing completely still":           "negative",
        "humanoid robot tumbling sideways":                   "negative",
    },
}

# Thresholds for flagging concerns in the formatted report.
# These are calibrated for softmax probabilities over all probes (~20-25 texts).
# A score of 0.45 on a single negative probe is very high given that baseline
# is ~1/N ≈ 0.04. Revisit if probe count changes significantly.
CONCERN_THRESHOLD  = 0.45   # individual negative probe score → CONCERN flag
HACKING_THRESHOLD  = 0.12   # negative_avg - positive_avg delta → HACKING warning
