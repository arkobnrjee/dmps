import gym


class SafeEnv(gym.Env):
    """ Base class for safe environments. """

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def safe(self):
        raise NotImplementedError
    
    def stable(self):
        raise NotImplementedError

    def backup(self):
        raise NotImplementedError
