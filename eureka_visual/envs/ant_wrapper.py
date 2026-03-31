"""
Gym wrapper and Observation Space documentation for Ant-v4.

OBSERVATION SPACE (27 dimensions):
Index 0: z-coordinate of the torso (height)
Index 1-4: orientation of the torso (quaternion x, y, z, w)
Index 5-12: joint angles for the 8 hinges (2 per leg)
Index 13: x-velocity of the torso
Index 14: y-velocity of the torso
Index 15: z-velocity of the torso
Index 16-18: angular velocity of the torso
Index 19-26: angular velocities of the 8 hinges

Actions (8 dimensions):
Torques applied to the 8 hinge joints.

Note:
When writing the reward function, ensure that you only reference valid indices.
Example: `obs[0]` for height, `obs[13]` and `obs[14]` for forward/lateral velocity.
"""

import gymnasium as gym

class AntKnowledgeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
