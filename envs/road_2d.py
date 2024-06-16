from .static_env import StaticEnv
import numpy as np
import gym


class Road2d(StaticEnv):
    def __init__(self):
        super().__init__(4, 2, np.array([-4., -4., -10., -10.]), np.array([4., 4., 10., 10.]), np.array([-2., -2.]), np.array([5., 5.]), 100)
        self.dt = 0.01
        self.kwargs = dict(ret_low=-1., ret_high=1., td3_unsafe_penalty=-20., eval_freq=4000)
        self.num_steps_to_train = 110000

    def _init_obs(self):
        return np.zeros(4)

    def _next_obs(self, obs, act):
        obs = obs.copy()
        obs[:2] += 10. * obs[2:] * self.dt
        obs[2:] += 5. * act * self.dt
        return obs

    def _reward(self, obs, act, next_obs):
        targ = np.array([3., 3.])
        return np.linalg.norm(obs[:2] - targ) - np.linalg.norm(next_obs[:2] - targ) if not self._done(next_obs) else 20.

    def _backup(self, obs):
        return (-obs[2:] / (5. * self.dt)).clip(-2., 5.)

    def _done(self, obs):
        return obs[0] >= 3. and obs[1] >= 3.

    def _safe(self, obs):
        return np.linalg.norm(obs[2:]) < 10.

    def _stable(self, obs):
        return self._safe(obs)
