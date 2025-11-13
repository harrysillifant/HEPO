import gymnasium as gym
from policy import HEPO
from custom_env import RewardSplitWrapper

env = gym.make("LunarLander-v3")
env = RewardSplitWrapper(env)


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
