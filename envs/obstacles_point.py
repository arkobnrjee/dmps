import numpy as np
from .safe_env import SafeEnv
import gym


N = 6
orbit_centers = np.array([[2 * np.cos(2 * np.pi * i / N), 2 * np.sin(2 * np.pi * i / N)] for i in range(N)])


class ObstaclesPoint(SafeEnv):
    def __init__(
            self, num_steps=500,
            obstacle_radius=.2, orbit_radius=0.85, orbit_centers=orbit_centers, obstacle_rate=6 * np.pi / 3,
            init_dist=4., max_act=1., dt=0.1,
            success_threshold=0.1, rew_factor=1., done_rew=10.,
            high_pos=5., max_vel=1.
        ):
        
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4 + 2 * N,))
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,))

        self.steps = 0
        self.num_steps = num_steps

        self.pos = None
        self.vel = None
        self.theta = None

        self.obstacle_radius = obstacle_radius
        self.orbit_radius = orbit_radius
        self.orbit_centers = orbit_centers
        self.num_obstacles = len(orbit_centers)
        self.init_dist = init_dist
        self.max_act = max_act
        self.dt = dt
        self.obstacle_rate = obstacle_rate
        self.success_threshold = success_threshold
        self.rew_factor = rew_factor
        self.done_rew = done_rew
        self.high_pos = high_pos
        self.max_vel = max_vel

        self.kwargs = dict(td3_unsafe_penalty=-12.)
        self.num_steps_to_train = 210000

    def reset(self):
        agent_theta = np.pi * (2. * np.random.rand() - 1.)
        self.pos = np.array([self.init_dist * np.cos(agent_theta), self.init_dist * np.sin(agent_theta)])
        self.vel = np.zeros(2)
        self.theta = np.pi * (2. * np.random.rand(self.num_obstacles) - 1.)
        self.steps = 0
        return self.obs()
    
    def step(self, act):
        act = act * self.max_act  # Unscale the action
        old_pos = self.pos.copy()

        self.pos = self.pos + self.vel * self.dt
        self.vel = (self.vel + act * self.dt).clip(-self.max_vel, self.max_vel)

        self.theta += self.obstacle_rate * self.dt
        self.theta -= (self.theta > np.pi) * 2 * np.pi
        self.theta += (self.theta < -np.pi) * 2 * np.pi

        terminated = np.linalg.norm(self.pos) < self.success_threshold

        rew = self.rew_factor * (np.linalg.norm(old_pos) - np.linalg.norm(self.pos)) - 0.01 if not terminated else self.done_rew

        self.steps += 1

        return self.obs(), rew, terminated or self.steps >= self.num_steps, {}

    def get_obstacles(self):
        obstacle_central_dir = np.stack((np.cos(self.theta), np.sin(self.theta)))
        return self.orbit_centers + self.orbit_radius * np.ndarray.transpose(obstacle_central_dir)
    
    def obs(self):
        obstacle_arr = np.ndarray.flatten(self.get_obstacles() - self.pos)
        obs = np.concatenate((self.pos / self.high_pos, self.vel / self.max_vel, obstacle_arr / self.high_pos))
        return obs

    def agent_state(self):
        return np.concatenate((self.pos, self.vel))
    
    def get_state(self):
        return self.pos.copy(), self.vel.copy(), self.theta.copy(), self.steps

    def set_state(self, state):
        self.pos, self.vel, self.theta, self.steps = state

    def safe(self):
        return np.min(np.sum(np.square(self.pos - self.get_obstacles()), axis=-1)) > self.obstacle_radius * self.obstacle_radius
    
    def stable(self):
        if np.max(np.abs(self.vel)) > 1e-3:
            return False

        dists_from_orbit_centers = np.sum(np.square(self.pos - self.orbit_centers), axis=-1)

        lower_dist = self.orbit_radius - self.obstacle_radius
        higher_dist = self.orbit_radius + self.obstacle_radius

        is_stable = np.all(np.logical_or(dists_from_orbit_centers < lower_dist * lower_dist, dists_from_orbit_centers > higher_dist * higher_dist))
        return is_stable
    
    def can_recover(self):
        if np.max(np.abs(self.vel)) > 1e-3:
            return True
        return self.stable()
    
    def backup(self):
        for center in self.orbit_centers:
            lower_dist = self.orbit_radius - self.obstacle_radius
            higher_dist = self.orbit_radius + self.obstacle_radius
            dist = np.linalg.norm(center - self.pos)
            if dist >= lower_dist and dist <= higher_dist:
                act = (self.pos) / self.dt
                act = act.clip(-self.max_act, self.max_act)
                return act / self.max_act
        act = -self.vel / self.dt
        act = act.clip(-self.max_act, self.max_act)
        return act / self.max_act
