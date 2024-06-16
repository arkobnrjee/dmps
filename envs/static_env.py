import numpy as np
import gym
from .safe_env import SafeEnv


class StaticEnv(SafeEnv):
    """ Base class for static environments. """

    def __init__(self, obs_dim, act_dim, obs_scale_low, obs_scale_high, act_scale_low, act_scale_high, num_steps):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,))
        self.action_space = gym.spaces.Box(-1., 1., shape=(act_dim,))

        self.obs_scale_low = obs_scale_low
        self.obs_scale_high = obs_scale_high
        self.act_scale_low = act_scale_low
        self.act_scale_high = act_scale_high

        self.steps = 0
        self.num_steps = num_steps

        self.obs = None

    def reset(self):
        self.obs = self._init_obs()
        self.steps = 0
        return self.scale_obs(self.obs)
    
    def step(self, act):
        act = self.unscale_act(act)
        next_obs = self._next_obs(self.obs, act)
        rew = self._reward(self.obs, act, next_obs)
        self.steps += 1
        done = self._done(next_obs) or self.steps >= self.num_steps
        self.obs = next_obs

        # We cap the observation to [-4., 4.].
        return self.scale_obs(self.obs).clip(-4., 4.), rew, done, {}

    def get_state(self):
        return self.obs.copy(), self.steps

    def set_state(self, state):
        self.obs, self.steps = state
        self.obs = self.obs.copy()

    def safe(self):
        return self._safe(self.obs)
    
    def stable(self):
        return self._stable(self.obs)

    def can_recover(self):
        return True
    
    def backup(self):
        return self.scale_act(self._backup(self.obs))

    def scale_obs(self, obs):
        obs = (obs - self.obs_scale_low) / (self.obs_scale_high - self.obs_scale_low)
        return 2 * obs - 1.

    def scale_act(self, act):
        act = (act - self.act_scale_low) / (self.act_scale_high - self.act_scale_low)
        return 2 * act - 1.
    
    def unscale_act(self, act):
        act = (act + 1.) / 2
        return act * (self.act_scale_high - self.act_scale_low) + self.act_scale_low

    def heuristic(self):
        return self._heuristic(self.obs)

    def _init_obs(self):
        raise NotImplementedError

    def _next_obs(self, obs, act):
        raise NotImplementedError

    def _reward(self, obs, act, next_obs):
        raise NotImplementedError

    def _backup(self, obs):
        raise NotImplementedError

    def _safe(self, obs):
        raise NotImplementedError
    
    def _done(self, obs):
        raise NotImplementedError

    def _stable(self, obs):
        raise NotImplementedError
    
    def _heuristic(self, obs):
        raise NotImplementedError
