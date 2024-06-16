from .static_env import StaticEnv
import numpy as np
import gym


class Obstacle2(StaticEnv):
    def __init__(self):
        super().__init__(4, 2, np.array([-0.5, -0.5, -5., -5.]), np.array([3.5, 3.5, 5., 5.]), -2. * np.ones(2), 5. * np.ones(2), 200)
        self.dt = 0.01
        self.kwargs = dict(ret_low=-1., ret_high=1., td3_unsafe_penalty=-30.)
        self.num_steps_to_train = 410000

    def _init_obs(self):
        return np.zeros(4)

    def _next_obs(self, obs, act):
        obs = obs.copy()
        obs[:2] += obs[2:] * self.dt
        obs[2:] += 2 * act * self.dt
        return obs

    def _reward(self, obs, act, next_obs):
        targ = np.array([3., 3.])
        return np.linalg.norm(obs[:2] - targ) - np.linalg.norm(next_obs[:2] - targ) if not self._done(next_obs) else 30.

    def _backup(self, obs):
        return (-obs[2:] / (2 * self.dt)).clip(-2., 5.)

    def _done(self, obs):
        return obs[0] >= 3 and obs[1] >= 3

    def _safe(self, obs):
        return obs[0] < 1 or obs[0] > 2 or obs[1] < 1 or obs[1] > 2

    def _stable(self, obs):
        return abs(obs[2]) < 1e-3 and abs(obs[3]) < 1e-3
