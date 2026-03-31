# probes/ant.py
PROBES = {
    "gait_pattern": {
        "quadruped diagonal trot gait":               "positive",
        "all four legs moving in coordination":       "positive",
        "only two legs on one side moving":           "negative",
        "legs moving in random uncoordinated order":  "negative",
    },
    "body_stability": {
        "body held level and stable while moving":    "positive",
        "body rolling or tilting to one side":        "negative",
        "body height consistent during locomotion":   "positive",
        "body bouncing excessively up and down":      "negative",
    },
    "contact": {
        "body dragging along the ground":             "negative",
        "all four feet making proper ground contact": "positive",
        "robot flipped upside down":                  "negative",
    },
    "failure_modes": {
        "robot tumbling or cartwheeling":             "negative",
        "robot spinning in circles":                  "negative",
        "robot stationary not making progress":       "negative",
    }
}
