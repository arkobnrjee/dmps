from .static_env import StaticEnv
import numpy as np
import gym


class Road(StaticEnv):
    def __init__(self):
        super().__init__(2, 1, np.array([-4., -10.]), np.array([4., 10.]), np.array([-2.]), np.array([5.]), 100)
        self.dt = 0.01
        self.kwargs = dict(ret_low=-1., ret_high=1., td3_unsafe_penalty=-20., eval_freq=4000)
        self.num_steps_to_train = 110000

    def _init_obs(self):
        return np.array([0., 0.])

    def _next_obs(self, obs, act):
        obs = obs.copy()
        obs[0] += 10. * obs[1] * self.dt
        obs[1] += 10. * act * self.dt
        return obs

    def _reward(self, obs, act, next_obs):
        return next_obs[0] - obs[0] if not self._done(next_obs) else 20.
        #return next_obs[0] - 3.

    def _backup(self, obs):
        return np.array([np.clip(-obs[1] / (10 * self.dt), -2., 5.)])

    def _done(self, obs):
        return obs[0] >= 3.

    def _safe(self, obs):
        return abs(obs[1]) < 10.

    def _stable(self, obs):
        return self._safe(obs)
