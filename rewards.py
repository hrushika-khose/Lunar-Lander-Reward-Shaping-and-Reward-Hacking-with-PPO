import gymnasium as gym

class TimePenalty(gym.Wrapper):
    """Subtract a small amount every frame."""
    def __init__(self, env, penalty=0.01):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = reward - self.penalty          # <-- our own 
        return obs, reward, terminated, truncated, info

class HoverBonus(gym.Wrapper):
    """BAD RULE: pays +1 every frame the rocket is upright and slow."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x, y, vx, vy, angle, spin, left_leg, right_leg = obs
        if abs(angle) < 0.2 and abs(vx) < 0.2 and abs(vy) < 0.2:
            reward += 1.0
        return obs, reward, terminated, truncated, info

class ImprovementBonus(gym.Wrapper):
    """GOOD RULE: reward getting better, charge a small fee per frame."""
    def __init__(self, env, fee=0.05):
        super().__init__(env)
        self.fee = fee
        self.prev_quality = 0.0

    def _quality(self, obs):
        x, y, vx, vy, angle, spin, l, r = obs
        calm = abs(angle) < 0.2 and abs(vx) < 0.2 and abs(vy) < 0.2
        return 1.0 if calm else 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_quality = self._quality(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        q = self._quality(obs)
        reward += (q - self.prev_quality)   # pay only for the CHANGE
        reward -= self.fee                  # waiting costs something
        self.prev_quality = q
        return obs, reward, terminated, truncated, info