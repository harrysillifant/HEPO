from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv

from hepo import HEPO
from custom_env import VectorizedRewardSplitWrapper
from predicates import LunarLander_predicate


if __name__ == "__main__":
    env1 = VectorizedRewardSplitWrapper(
        make_vec_env("LunarLander-v3", n_envs=4), task_predicate=None
    )
    env2 = VectorizedRewardSplitWrapper(
        make_vec_env("LunarLander-v3", n_envs=4), task_predicate=None
    )
    #
    # eval_env = VectorizedRewardSplitWrapper(
    #     make_vec_env("LunarLander-v3", n_envs=1))
    #
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path="./logs",
    #     log_path="./logs",
    #     eval_freq=100,
    #     deterministic=True,
    #     render=True,
    # )
    #
    model = HEPO(
        "MlpPolicy", env1=env1, env2=env2, tensorboard_log="./hepovsppo6_tb_logs/"
    )

    model.learn(total_timesteps=10_000, progress_bar=False)
