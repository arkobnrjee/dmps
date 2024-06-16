from envs import get_env
import argparse
import numpy as np
import torch
from copy import deepcopy
import logging


class Normal:
    def __init__(self, size):
        self.size = size
    def gen(self):
        return np.random.normal(0., 0.2, size=self.size)

class OU:
    def __init__(self, size, sigma=0.2):
        self.theta = 0.15
        self.size = size
        self.mu = np.zeros(size)
        self.sigma = sigma * np.ones(size)
        self.dt = 1e-2
        self.prev = np.zeros(size)

    def gen(self):
        self.prev = (
            self.prev 
            + self.theta * (self.mu - self.prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        )
        return self.prev


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="GatePoint")
    parser.add_argument("--mode", type=str, default="dmps")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--log_to", type=str)
    parser.add_argument("--save_to", type=str)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = get_env(args.env)
    test_env = get_env(args.env)
    env.action_space.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}") if args.gpu is not None else torch.device("cpu")

    kwargs = deepcopy(env.kwargs)
    kwargs.update(eval_log_file=args.log_to, model_save_dir=args.save_to, act_noise=Normal(env.action_space.shape[-1]))

    if args.env in ["MountainCar", "Obstacle", "Obstacle2", "Road", "Road2d", "Pendulum"]:
        sigma = 0.5 if args.env == "MountainCar" else 0.2
        kwargs.update(act_noise=OU(env.action_space.shape[-1], sigma=sigma))

    if "td3_unsafe_penalty" in kwargs and args.mode == "td3":
        kwargs["unsafe_penalty"] = kwargs["td3_unsafe_penalty"]

    kwargs.pop("td3_unsafe_penalty", None)

    if args.mode == "dmps":
        from mcts_trainer import MCTSTrainer
        trainer = MCTSTrainer(env, test_env, device, **kwargs)
    elif args.mode == "mps":
        from mps_trainer import MPSTrainer
        trainer = MPSTrainer(env, test_env, device, **kwargs)
    else:
        from td3_trainer import Trainer
        trainer = Trainer(env, test_env, device, **kwargs)

    trainer.run(env.num_steps_to_train)
