import numpy as np
from .safe_env import SafeEnv
import gym


class GatePoint(SafeEnv):
    def __init__(
            self, num_steps=500,
            inner_radius=1., outer_radius=1.1, init_dist=2., max_act=1., dt=0.1,
            obstacle_rate=2 * np.pi / 3, success_threshold=0.1, rew_factor=1., done_rew=10.,
            high_pos=4., max_vel=1., obstacle_opening_theta=np.pi / 3
        ):

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(6,))
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,))

        self.steps = 0
        self.num_steps = num_steps

        self.pos = None
        self.vel = None
        self.theta = None

        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.init_dist = init_dist
        self.max_act = max_act
        self.dt = dt
        self.obstacle_rate = obstacle_rate
        self.success_threshold = success_threshold
        self.rew_factor = rew_factor
        self.done_rew = done_rew
        self.high_pos = high_pos
        self.max_vel = max_vel
        self.obstacle_opening_theta = obstacle_opening_theta

        self.kwargs = dict(td3_unsafe_penalty=-12.)
        self.num_steps_to_train = 210000

    def reset(self):
        agent_theta = np.pi * (2. * np.random.rand() - 1.)
        self.pos = np.array([self.init_dist * np.cos(agent_theta), self.init_dist * np.sin(agent_theta)])
        self.vel = np.zeros(2)
        self.theta = np.pi * (2. * np.random.rand() - 1.)
        self.steps = 0
        return self.obs()
    
    def step(self, act):
        act = act * self.max_act  # Unscale the action
        old_pos = self.pos.copy()

        self.pos = self.pos + self.vel * self.dt
        self.vel = (self.vel + act * self.dt).clip(-self.max_vel, self.max_vel)

        self.theta += self.obstacle_rate * self.dt
        # This assumes the rate isn't abnormal
        if self.theta > np.pi:
            self.theta -= 2 * np.pi
        if self.theta < -np.pi:
            self.theta += 2 * np.pi

        terminated = np.linalg.norm(self.pos) < self.success_threshold

        rew = self.rew_factor * (np.linalg.norm(old_pos) - np.linalg.norm(self.pos)) - 0.005 if not terminated else self.done_rew

        self.steps += 1

        return self.obs(), rew, terminated or self.steps >= self.num_steps, {}
    
    def obs(self):
        theta_arr = np.array([np.cos(self.theta), np.sin(self.theta)])
        obs = np.concatenate((self.pos / self.high_pos, self.vel / self.max_vel, theta_arr))
        return obs

    def agent_state(self):
        return np.concatenate((self.pos, self.vel))
    
    def get_state(self):
        return self.pos.copy(), self.vel.copy(), self.theta, self.steps

    def set_state(self, state):
        self.pos, self.vel, self.theta, self.steps = state

    def safe(self):
        dist = np.linalg.norm(self.pos)
        if dist < self.inner_radius or dist > self.outer_radius:
            return True
        
        theta = np.arctan2(self.pos[1], self.pos[0])

        diff = abs(theta - self.theta)
        return diff < self.obstacle_opening_theta or diff > 2 * np.pi - self.obstacle_opening_theta
    
    def stable(self):
        if np.max(np.abs(self.vel)) > 1e-3:
            return False

        dist = np.linalg.norm(self.pos)
        is_stable = dist < self.inner_radius or dist > self.outer_radius
        return is_stable

    def can_recover(self):
        if np.max(np.abs(self.vel)) > 1e-3:
            return True
        return self.stable()
    
    def backup(self):
        dist = np.linalg.norm(self.pos)
        if dist > self.inner_radius and dist < self.outer_radius:
            act = self.pos / self.dt
            act = act.clip(-self.max_act, self.max_act)
            return act / self.max_act
        act = -self.vel / self.dt
        act = act.clip(-self.max_act, self.max_act)
        return act / self.max_act
