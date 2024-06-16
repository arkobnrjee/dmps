from .static_env import StaticEnv
import numpy as np
import gym


class MountainCar(StaticEnv):
    def __init__(self):
        super().__init__(2, 1, np.array([-1.2, -0.007]), np.array([0.7, 0.007]), np.array([-1.]), np.array([1.]), 999)
        self.dt = 1.
        self.kwargs = dict(unsafe_penalty=-1.)
        self.num_steps_to_train = 210000

    def _init_obs(self):
        return np.array([-0.5, 0.])

    def _next_obs(self, obs, act):
        obs = obs.copy()
        delta_obs = np.array([obs[1], 0.001 * act[0] - 0.0025 * np.cos(3 * obs[0])])
        obs += delta_obs * self.dt
        return obs

    def _reward(self, obs, act, next_obs):
        return 100. if self._done(next_obs) else -0.1 * np.linalg.norm(act) ** 2

    def _backup(self, obs):
        return np.array([1.])

    def _done(self, obs):
        return obs[0] >= 0.6

    def _safe(self, obs):
        return obs[0] > -np.pi / 3

    def _stable(self, obs):
        return obs[1] >= 0.
