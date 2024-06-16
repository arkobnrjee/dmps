import numpy as np
from .safe_env import SafeEnv
import gym


class GateCar(SafeEnv):
    def __init__(
            self, num_steps=500,
            inner_radius=1., outer_radius=1.1, init_dist=2., max_act=1., dt=0.1,
            obstacle_rate=2 * np.pi / 3, success_threshold=0.1, rew_factor=1., done_rew=10.,
            high_pos=4., max_vel=1., max_angle_vel=2 * np.pi, obstacle_opening_theta=np.pi / 3
        ):

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,))
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,))

        self.steps = 0
        self.num_steps = num_steps

        self.state = None

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
        self.max_angle_vel = max_angle_vel
        self.obstacle_opening_theta = obstacle_opening_theta

        self.kwargs = dict(td3_unsafe_penalty=-12.)
        self.num_steps_to_train = 210000

    def reset(self):
        agent_theta = np.pi * (2. * np.random.rand() - 1.)
        agent_angle = np.pi * (2. * np.random.rand() - 1.)

        self.state = np.array([self.init_dist * np.cos(agent_theta), self.init_dist * np.sin(agent_theta), agent_angle, 0., 0.])

        self.theta = np.pi * (2. * np.random.rand() - 1.)
        self.steps = 0
        return self.obs()
    
    def step(self, act):
        act = act * self.max_act  # Unscale the action
        old_pos = self.state[:2].copy()

        self.state = self._next_state(self.state, act)

        self.theta += self.obstacle_rate * self.dt

        # This assumes the rate isn't abnormal
        if self.theta > np.pi:
            self.theta -= 2 * np.pi
        if self.theta < -np.pi:
            self.theta += 2 * np.pi

        pos = self.state[:2]
        terminated = np.linalg.norm(pos) < self.success_threshold

        rew = self.rew_factor * (np.linalg.norm(old_pos) - np.linalg.norm(pos)) - 0.005 if not terminated else self.done_rew

        self.steps += 1

        return self.obs(), rew, terminated or self.steps >= self.num_steps, {}
    
    def _next_state(self, state, act):
        x, y, angle, v, angle_v = state
        delta = np.array([v * np.cos(angle), v * np.sin(angle), angle_v, 1 / 2 * (act[0] + act[1]), 3 * (act[0] - act[1])])
        next_state = state + delta * self.dt
        next_state[3] = np.clip(next_state[3], -self.max_vel, self.max_vel)
        next_state[4] = np.clip(next_state[4], -self.max_angle_vel, self.max_angle_vel)
        return next_state

    def obs(self):
        # Let's decompose the state
        x, y, angle, v, angle_v = self.state

        obs = np.array(
            [x / self.high_pos, y / self.high_pos,
             np.cos(angle), np.sin(angle), v / self.max_vel,
             angle_v / self.max_angle_vel, np.cos(self.theta), np.sin(self.theta)
             ])
        
        return obs
    
    def agent_state(self):
        return self.state
    
    def get_state(self):
        return self.state.copy(), self.theta, self.steps

    def set_state(self, state):
        self.state, self.theta, self.steps = state

    def safe(self):
        dist = np.linalg.norm(self.state[:2])
        if dist < self.inner_radius or dist > self.outer_radius:
            return True
        
        theta = np.arctan2(self.state[1], self.state[0])

        diff = abs(theta - self.theta)
        return diff < self.obstacle_opening_theta or diff > 2 * np.pi - self.obstacle_opening_theta
    
    def stable(self):
        if abs(self.state[3]) > 1e-3:
            return False

        dist = np.linalg.norm(self.state[:2])
        is_stable = dist < self.inner_radius or dist > self.outer_radius
        return is_stable

    def can_recover(self):
        if abs(self.state[3]) > 1e-3:
            return True
        return self.stable()
    
    def backup(self):
        dist = np.linalg.norm(self.state[:2])
        if dist > self.inner_radius and dist < self.outer_radius:
            return -np.ones(2)
        act = -self.state[3] / self.dt * np.ones(2)
        act = act.clip(-self.max_act, self.max_act)
        return act / self.max_act
