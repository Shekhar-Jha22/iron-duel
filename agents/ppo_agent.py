import numpy as np
from observations import to_world_action

class PPOAgent:
    """PPO agent that can load trained models or act randomly as fallback"""
    name = "PPO"
    color = "#22c55e"
    uses_obs = True

    def __init__(self, model=None, deterministic=False, team=0):
        self.model = model
        self.deterministic = deterministic
        self.fallback_enabled = False
        self.team = team

    def get_action(self, obs, env_state):
        """
        Get action from PPO model or random fallback
        
        Args:
            obs: observation from environment
            env_state: full game state
            
        Returns:
            action: [move (0-8), aim (0-15), shoot (0-1)]
        """
        if self.model is None:
            if self.fallback_enabled:
                return self._random_action()
            else:
                raise ValueError("PPO model not loaded and fallback disabled")
        
        try:
            action, _states = self.model.predict(obs, deterministic=self.deterministic)
            return to_world_action(action, self.team)
        except Exception as e:
            print(f"⚠ PPO prediction failed: {e}, falling back to random")
            return self._random_action()

    def _random_action(self):
        """Fallback random action"""
        return np.array([
            np.random.randint(0, 9),   # movement (8 directions + stay)
            np.random.randint(0, 16),  # aim (16 angles)
            np.random.randint(0, 2),   # shoot (0 or 1)
        ], dtype=np.int64)

    def load(self, path):
        """Load trained PPO model from path"""
        from stable_baselines3 import PPO as PPOModel   # training-only dependency
        try:
            self.model = PPOModel.load(path)
            print(f"✓ Loaded PPO model from {path}")
        except FileNotFoundError:
            print(f"✗ Model not found at {path}, using random fallback")
            self.model = None

    def save(self, path):
        """Save PPO model to path"""
        if self.model is not None:
            self.model.save(path)
            print(f"✓ Saved PPO model to {path}")
        else:
            print("✗ No model to save")