import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.schedules import FloatSchedule
from utils import HEPO

vec_env = make_vec_env("LunarLander-v3", n_envs=4)


"""
Steps: (for each iteration)
Rollout B/2 trajectories for pi add to trajectory buffer for pi
Rollout B/2 trajectories for pi_ref add to trajectory buffer for pi_ref
Compute advantages for pi on task and heuristic reward using GAE with the value fns for pi on task and heuristic
Compute advantages for pi_H on task and heuristic reward using GAE with the value fns for pi_H on task and heuristic
Compute the surrogate task objectives for pi and pi_H using advantage fns computed above along with lagrange multiplier alpha
Compute the surrogate heuristic objectives for pi and pi_H using advantage fns computed above along with lagrange multiplier alpha

Update pi based on surrogate task objectives
Update pi_H based on surrogate heuristic objectives
Update the 4 value functions
Update alpha
"""
