"""
CLIP probe bank for Ant-v4 quadruped locomotion.
Same structural conventions as humanoid.py.
"""

ROBOT_TYPE = "ant"
ENV_ID = "Ant-v4"

PROBES = {
    "gait_pattern": {
        "quadruped robot with diagonal trot gait pattern":               "positive",
        "quadruped robot with all four legs coordinated":                "positive",
        "quadruped robot with natural four-legged walking stride":       "positive",
        "quadruped robot with only two legs on same side moving":        "negative",
        "quadruped robot with legs moving in random uncoordinated way":  "negative",
        "quadruped robot legs all moving at same time":                  "negative",
    },
    "body_stability": {
        "quadruped robot body held level and stable while moving":   "positive",
        "quadruped robot consistent body height during locomotion":  "positive",
        "quadruped robot body rolling or tilting sideways":          "negative",
        "quadruped robot body bouncing excessively up and down":     "negative",
        "quadruped robot body swinging side to side":                "negative",
    },
    "foot_contact": {
        "quadruped robot with all four feet making proper ground contact":  "positive",
        "quadruped robot with feet pressing firmly into the ground":        "positive",
        "quadruped robot body dragging along the ground surface":           "negative",
        "quadruped robot with feet slipping without traction":              "negative",
    },
    "forward_progress": {
        "quadruped robot moving steadily forward":                      "positive",
        "quadruped robot with consistent forward velocity":             "positive",
        "quadruped robot spinning in circles instead of going forward": "negative",
        "quadruped robot moving sideways instead of forward":           "negative",
        "quadruped robot stationary not making forward progress":       "negative",
    },
    "failure_modes": {
        "quadruped robot flipped upside down":       "negative",
        "quadruped robot tumbling or cartwheeling":  "negative",
        "quadruped robot fallen on its side":        "negative",
        "quadruped robot collapsed on the ground":   "negative",
    },
}

# Same calibration rationale as humanoid.py
CONCERN_THRESHOLD  = 0.45
HACKING_THRESHOLD  = 0.12
