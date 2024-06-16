import numpy as np
from .safe_env import SafeEnv
import gym


N = 6
orbit_centers = np.array([[2 * np.cos(2 * np.pi * i / N), 2 * np.sin(2 * np.pi * i / N)] for i in range(N)])


class ObstaclesCar(SafeEnv):
    def __init__(
            self, num_steps=500,
            obstacle_radius=0.2, orbit_radius=0.85, orbit_centers=orbit_centers, obstacle_rate=2 * np.pi,
            init_dist=4., max_act=1., dt=0.1,
            success_threshold=0.1, rew_factor=1., done_rew=10.,
            high_pos=4., max_vel=1., max_angle_vel=2 * np.pi
        ):
        
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(6 + 2 * N,))
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,))

        self.steps = 0
        self.num_steps = num_steps

        self.state = None
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
        self.max_angle_vel = max_angle_vel

        self.kwargs = dict(td3_unsafe_penalty=-12.)
        self.num_steps_to_train = 210000

    def reset(self):
        agent_theta = np.pi * (2. * np.random.rand() - 1.)
        agent_angle = np.pi * (2. * np.random.rand() - 1.)

        self.state = np.array([self.init_dist * np.cos(agent_theta), self.init_dist * np.sin(agent_theta), agent_angle, 0., 0.])

        self.theta = np.pi * (2. * np.random.rand(self.num_obstacles) - 1.)
        self.steps = 0
        return self.obs()
    
    def step(self, act):
        act = act * self.max_act  # Unscale the action
        old_pos = self.state[:2].copy()

        self.state = self._next_state(self.state, act)

        self.theta += self.obstacle_rate * self.dt
        self.theta -= (self.theta > np.pi) * 2 * np.pi
        self.theta += (self.theta < -np.pi) * 2 * np.pi

        self.theta += self.obstacle_rate * self.dt

        pos = self.state[:2]
        terminated = np.linalg.norm(pos) < self.success_threshold

        rew = self.rew_factor * (np.linalg.norm(old_pos) - np.linalg.norm(pos)) - 0.01 if not terminated else self.done_rew

        self.steps += 1

        return self.obs(), rew, terminated or self.steps >= self.num_steps, {}
    
    def _next_state(self, state, act):
        x, y, angle, v, angle_v = state
        delta = np.array([v * np.cos(angle), v * np.sin(angle), angle_v, 1 / 2 * (act[0] + act[1]), 3 * (act[0] - act[1])])
        next_state = state + delta * self.dt
        next_state[3] = np.clip(next_state[3], -self.max_vel, self.max_vel)
        next_state[4] = np.clip(next_state[4], -self.max_angle_vel, self.max_angle_vel)
        return next_state

    def get_obstacles(self):
        obstacle_central_dir = np.stack((np.cos(self.theta), np.sin(self.theta)))
        return self.orbit_centers + self.orbit_radius * np.ndarray.transpose(obstacle_central_dir)

    def obs(self):
        # Let's decompose the state
        obstacle_arr = np.ndarray.flatten(self.get_obstacles() - self.state[:2])
        x, y, angle, v, angle_v = self.state

        obs = np.array(
            [x / self.high_pos, y / self.high_pos,
             np.cos(angle), np.sin(angle), v / self.max_vel,
             angle_v / self.max_angle_vel])

        obs = np.concatenate((obs, obstacle_arr / self.high_pos))
        
        return obs
    
    def agent_state(self):
        return self.state
    
    def get_state(self):
        return self.state.copy(), self.theta.copy(), self.steps

    def set_state(self, state):
        self.state, self.theta, self.steps = state

    def safe(self):
        pos = self.state[:2]
        return np.min(np.sum(np.square(pos - self.get_obstacles()), axis=-1)) > self.obstacle_radius * self.obstacle_radius
    
    def stable(self):
        if abs(self.state[3]) > 1e-3:
            return False

        pos = self.state[:2]

        dists_from_orbit_centers = np.sum(np.square(pos - self.orbit_centers), axis=-1)

        lower_dist = self.orbit_radius - self.obstacle_radius
        higher_dist = self.orbit_radius + self.obstacle_radius

        is_stable = np.all(np.logical_or(dists_from_orbit_centers < lower_dist * lower_dist, dists_from_orbit_centers > higher_dist * higher_dist))
        return is_stable
    
    def can_recover(self):
        if abs(self.state[3]) > 1e-3:
            return True 
        return self.stable()
    
    def backup(self):
        pos = self.state[:2]

        dists_from_orbit_centers = np.sum(np.square(pos - self.orbit_centers), axis=-1)

        lower_dist = self.orbit_radius - self.obstacle_radius
        higher_dist = self.orbit_radius + self.obstacle_radius

        is_stable = np.all(np.logical_or(dists_from_orbit_centers < lower_dist * lower_dist, dists_from_orbit_centers > higher_dist * higher_dist))
        if not is_stable:
            if self.state[4] > 0:
                return np.array([-1., -0.3])
            else:
                return np.array([-0.3, -1.])
            return -np.ones(2)

        act = -self.state[3] / self.dt * np.ones(2)
        act = act.clip(-self.max_act, self.max_act)
        return act / self.max_act
