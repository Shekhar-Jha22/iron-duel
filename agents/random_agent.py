import numpy as np

class RandomAgent:

    name = "Random"
    color = "#6b7280"

    def __init__(self, team=0):
        self.team = team

    def get_action(self, obs, env_state):
        return np.array([
            np.random.randint(0, 9),
            np.random.randint(0, 16),
            np.random.randint(0, 2),
        ], dtype=np.int64)