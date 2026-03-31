# probes/humanoid.py
PROBES = {
    "posture": {
        "upright balanced torso":                    "positive",
        "torso bent far forward while moving":        "negative",
        "robot crouching low to the ground":          "negative",
        "humanoid with straight vertical spine":      "positive",
    },
    "gait_symmetry": {
        "symmetric alternating leg movement":         "positive",
        "hopping on one leg repeatedly":              "negative",
        "both legs moving together like a jump":      "negative",
        "natural human walking stride pattern":       "positive",
    },
    "stability": {
        "smooth continuous forward motion":           "positive",
        "jerky oscillating unstable movement":        "negative",
        "robot stumbling and recovering":             "negative",
        "steady consistent velocity":                 "positive",
    },
    "naturalness": {
        "arms swinging in opposition to legs":        "positive",
        "arms rigid and fixed while walking":         "negative",
        "stiff mechanical unnatural movement":        "negative",
        "fluid natural human-like locomotion":        "positive",
    },
    "foot_contact": {
        "feet lifting cleanly off the ground":        "positive",
        "feet dragging or shuffling along floor":     "negative",
        "robot sliding without proper foot contact":  "negative",
    },
    "failure_modes": {
        "robot falling over sideways":                "negative",
        "robot spinning in place":                    "negative",
        "robot moving backwards":                     "negative",
        "robot standing still not moving":            "negative",
    }
}
