import numpy as np
import torch
from torch.optim import Adam
from models import Model
from utils import polyak_update
import math
import logging
import os


class ReplayBuffer:
    def __init__(self, max_size, obs_dim, act_dim):
        self.size = 0
        self.idx = 0
        self.max_size = max_size
        self.obs_buf = np.empty((max_size, obs_dim))
        self.act_buf = np.empty((max_size,  act_dim))
        self.next_obs_buf = np.empty((max_size, obs_dim))
        self.rew_buf = np.empty(max_size)
        self.done_buf = np.empty(max_size)


    def add(self, obs, act, next_obs, rew, done):
        self.obs_buf[self.idx] = obs
        self.act_buf[self.idx] = act
        self.next_obs_buf[self.idx] = next_obs
        self.rew_buf[self.idx] = rew
        self.done_buf[self.idx] = done

        if self.size < self.max_size:
            self.size += 1

        self.idx += 1
        if self.idx == self.max_size:
            self.idx = 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.obs_buf[idxs], self.act_buf[idxs], self.next_obs_buf[idxs],
                self.rew_buf[idxs], self.done_buf[idxs])


class Trainer:
    def __init__(self, env, test_env, device,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 replay_buffer_size=int(1e6),
                 ret_low=-1.,
                 ret_high=1.,
                 stop_at_unsafe=False,
                 unsafe_penalty=-0.1,
                 gamma=0.99,
                 train_interval=250,
                 num_grad_steps=-1,
                 batch_size=256,
                 tau=0.005,
                 warmup_steps=int(25e3),
                 act_noise=None,
                 policy_delay=2,
                 targ_policy_noise=0.2,
                 targ_noise_clip=0.5,
                 model_save_freq=int(2e5),
                 model_save_dir=None,
                 eval_freq=int(1e4),
                 eval_num_episodes=10,
                 eval_log_file=None):

        self.env = env
        self.test_env = test_env
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.stop_at_unsafe = stop_at_unsafe
        self.unsafe_penalty = unsafe_penalty
        self.gamma = gamma
        self.train_interval = train_interval
        self.num_grad_steps = train_interval if num_grad_steps == -1 else num_grad_steps
        self.batch_size = batch_size
        self.tau = tau
        self.warmup_steps = max(warmup_steps, batch_size)
        self.act_noise = act_noise
        self.policy_delay = policy_delay
        self.targ_policy_noise = targ_policy_noise
        self.targ_noise_clip = targ_noise_clip
        self.model_save_freq = model_save_freq
        self.model_save_dir = model_save_dir
        self.eval_freq = eval_freq
        self.eval_num_episodes = eval_num_episodes
        self.eval_log_file = eval_log_file

        self.max_ep_len = env.num_steps
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.model = Model(self.obs_dim, self.act_dim, device, ret_low=ret_low, ret_high=ret_high)
        self.actor_opt = Adam(self.model.actor.parameters(), lr=actor_lr)
        self.critic1_opt = Adam(self.model.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = Adam(self.model.critic2.parameters(), lr=critic_lr)

        self.buf = ReplayBuffer(replay_buffer_size, self.obs_dim, self.act_dim)

        self.num_updates = 0

        if eval_log_file is not None:
            if os.path.dirname(eval_log_file) != "":
                os.makedirs(os.path.dirname(eval_log_file), exist_ok=True)
            logging.basicConfig(filename=eval_log_file, level=logging.INFO)

        if model_save_dir is not None:
            os.makedirs(model_save_dir, exist_ok=True)

    def run(self, num_timesteps):
        # These are stats for the current runthrough
        ep_num = 1
        ep_len = 0
        total_rew = 0.
        num_safety_violations = 0

        obs = self.env.reset()

        for t in range(num_timesteps):
            if t < self.warmup_steps:
                act = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    act = self.model.actor(torch.as_tensor(obs, dtype=torch.float32, device=self.device)).cpu().numpy()

                # Add noise to the action
                if self.act_noise is not None:
                    act = (act + self.act_noise.gen()).clip(-1., 1.)

            next_obs, rew, done, _ = self.env.step(act)
            ep_len += 1
            total_rew += rew

            if not self.env.safe():
                rew = self.unsafe_penalty
                num_safety_violations += 1
                if self.stop_at_unsafe:
                    done = True

            terminated = done and ep_len < self.max_ep_len
            self.buf.add(obs, act, next_obs, rew, done)
            
            obs = next_obs

            if done:
                # Print stats about this run
                result = f"T: {t + 1}\tEp: {ep_num}\tRew: {total_rew}\tUnsafe: {num_safety_violations}\tLen: {ep_len}"

                if self.eval_log_file is not None:
                    logging.info(result)

                ep_num += 1
                ep_len = 0
                total_rew = 0.

                obs = self.env.reset()

            if t > self.warmup_steps and t % self.eval_freq == 0:
                self.evaluate(t, ep_num)

            if t > self.warmup_steps and t % self.train_interval == 0:
                self.train(self.num_grad_steps)

            if t > self.warmup_steps and t % self.model_save_freq == 0 and self.model_save_dir is not None:
                torch.save(self.model, self.model_save_dir + "/save_" + str(t) + ".pth")

        return self.model

    def train(self, num_steps):
        for _ in range(num_steps):
            obs_batch, act_batch, next_obs_batch, rew_batch, done_batch = self.buf.sample(batch_size=self.batch_size)

            # Tensorify
            obs_batch = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
            act_batch = torch.as_tensor(act_batch, dtype=torch.float32, device=self.device)
            next_obs_batch = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
            rew_batch = torch.as_tensor(rew_batch, dtype=torch.float32, device=self.device)
            done_batch = torch.as_tensor(done_batch, dtype=torch.float32, device=self.device)

            # We first need to compute q_targ
            with torch.no_grad():
                policy_targ = self.model.actor_targ(next_obs_batch)
                policy_targ_noise = torch.normal(
                    torch.zeros(self.batch_size, self.act_dim, device=self.device), self.targ_policy_noise
                )

                policy_targ_noise = policy_targ_noise.clip(-self.targ_noise_clip, self.targ_noise_clip)
                policy_targ_with_noise = policy_targ + policy_targ_noise

                policy_targ_with_noise = policy_targ_with_noise.clip(-1., 1.)

                q_targ = (rew_batch
                          + self.gamma * (1. - done_batch)
                              * self.model.critic_targ(next_obs_batch, policy_targ_with_noise).squeeze(-1))

            # Update critic 1
            q1_curr = self.model.critic1(obs_batch, act_batch).squeeze(-1)
            critic1_loss = (q1_curr - q_targ).square().mean()

            self.critic1_opt.zero_grad()
            critic1_loss.backward()
            self.critic1_opt.step()

            # Update critic 2
            q2_curr = self.model.critic2(obs_batch, act_batch).squeeze(-1)
            critic2_loss = (q2_curr - q_targ).square().mean()

            self.critic2_opt.zero_grad()
            critic2_loss.backward()
            self.critic2_opt.step()

            if self.num_updates % self.policy_delay == 0:
                loss_pi = -self.model.critic1(obs_batch, self.model.actor(obs_batch)).mean()
                self.actor_opt.zero_grad()
                loss_pi.backward()
                self.actor_opt.step()

                polyak_update(self.model.actor.parameters(), self.model.actor_targ.parameters(), self.tau)
                polyak_update(self.model.critic1.parameters(), self.model.critic1_targ.parameters(), self.tau)
                polyak_update(self.model.critic2.parameters(), self.model.critic2_targ.parameters(), self.tau)

            self.num_updates += 1

    def evaluate(self, timestep, episode):
        rewards = []
        safety_violations = []
        ep_len = []
        
        for _ in range(self.eval_num_episodes):
            obs = self.test_env.reset()
            done = False
            cumulative_rew = 0.
            cumulative_violations = 0
            num_steps = 0

            while not done:
                with torch.no_grad():
                    act = self.model.actor(torch.as_tensor(obs, dtype=torch.float32, device=self.device)).cpu().numpy()
                obs, rew, done, _ = self.test_env.step(act)
                cumulative_rew += rew
                if not self.test_env.safe():
                    cumulative_violations += 1
                    if self.stop_at_unsafe:
                        done = True
                num_steps += 1

            rewards.append(cumulative_rew)
            safety_violations.append(cumulative_violations)
            ep_len.append(num_steps)
            logging.info(f"EVAL: {timestep}\tEp: {episode}\tRew: {cumulative_rew}\t"
                         f"Unafe: {cumulative_violations}\tLen: {num_steps}\t\n")

        logging.info(f"EVALAVG: {timestep}\tEp: {episode}\tRewAvg: {np.mean(rewards)}\tRewStd: {np.std(rewards)}\t"
                     f"Unsafe: {np.mean(safety_violations)}\tLen: {np.mean(ep_len)}\n")
