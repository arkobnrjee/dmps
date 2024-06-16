import numpy as np
import torch


class MPSPlanner:
    def __init__(self, env):
        self.env = env
        self.obs = None
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def take_step(self, base_act):
        if self.check_action(base_act):
            self.obs, rew, done, info = self.env.step(base_act)
            return [(self.obs, rew, done, info)], ([base_act], 0)

        backup_act = self.env.backup()
        self.obs, rew, done, info = self.env.step(backup_act)
        return [(self.obs, rew, done, info)], ([backup_act], 1)

    def check_action(self, action):
        state = self.env.get_state()
        is_recoverable = self.check_action_internal(action)
        self.env.set_state(state)
        return is_recoverable

    def check_action_internal(self, action):
        self.env.step(action)
        if not self.env.safe():
            return False

        while True:
            backup_act = self.env.backup()
            self.env.step(backup_act)
            if not self.env.safe():
                return False
            if self.env.stable():
                return True
