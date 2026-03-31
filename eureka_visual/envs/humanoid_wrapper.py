"""
Gym wrapper and Observation Space documentation for Humanoid-v4.

OBSERVATION SPACE (376 dimensions):
Index 0: z-coordinate of the torso (height)
Index 1-4: orientation of the torso (quaternion x, y, z, w)
Index 5-15: joint angles (abdomen, legs, arms)
Index 16-22: center of mass velocity (xyz over time)
Index 23-36: angular velocities of joints
Index 37-375: center of mass based inertia and various external forces/contacts.

Actions (17 dimensions):
Torques applied to 17 hinge joints.

Note:
When writing the reward function, ensure that you only reference valid indices. 
Example: `obs[0]` for height, `obs[1:5]` for quaternion orientation.

CRITICAL PHYSICS BOUNDARIES:
- The standard starting height for `obs[0]` is roughly 1.4. 
- The environment automatically TERMINATES the episode if `obs[0]` falls below 1.0 or goes above 2.0. 
- NEVER reward the robot for targeting a height below 1.0, or episodes will end instantly!
"""

import gymnasium as gym

class HumanoidKnowledgeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
