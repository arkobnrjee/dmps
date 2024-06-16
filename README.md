# Dynamic Model Predictive Shielding for Provably Safe Reinforcement Learning

Implementation code for paper: https://arxiv.org/abs/2405.13863

This repository contains the code for DMPS, MPS, and TD3.

Dependencies are NumPy, PyTorch, and OpenAI Gym.

The main evaluation module is in `main.py`. It can be run with
mode options of dmps, mps, or td3.

To run all benchmarks automatically, run `python dispatch.py`.
This will put all results in a directory labelled `results`.

Implementations of the environments are in the `envs` directory.
