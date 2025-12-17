import gymnasium as gym
from hepo import HEPO
from custom_env import RewardSplitWrapper, VectorizedRewardSplitWrapper
from stable_baselines3.common.env_util import make_vec_env


if __name__ == "__main__":
    env1 = VectorizedRewardSplitWrapper(
        make_vec_env("LunarLander-v3", n_envs=4))
    env2 = VectorizedRewardSplitWrapper(
        make_vec_env("LunarLander-v3", n_envs=4))

    model = HEPO("MlpPolicy", env1=env1, env2=env2)

    model.learn(total_timesteps=200_000)
