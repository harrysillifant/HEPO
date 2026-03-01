from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv

from hepo.hepo import HEPO
from hepo.custom_env import VecEnvRewardSplitWrapper
from hepo.predicates import LunarLander_predicate, BipedalWalker_predicate


if __name__ == "__main__":
    # env1 = VecEnvRewardSplitWrapper(
    #     make_vec_env("LunarLander-v3", n_envs=4),
    #     task_predicate=LunarLander_predicate(),
    # )
    # env2 = VecEnvRewardSplitWrapper(
    #     make_vec_env("LunarLander-v3", n_envs=4),
    #     task_predicate=LunarLander_predicate(),
    # )
    env1 = VecEnvRewardSplitWrapper(
        make_vec_env("CartPole-v1", n_envs=4),
        task_predicate=BipedalWalker_predicate(),
    )
    env2 = VecEnvRewardSplitWrapper(
        make_vec_env("CartPole-v1", n_envs=4),
        task_predicate=BipedalWalker_predicate(),
    )

    # eval_env = VecEnvRewardSplitWrapper(
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
        "MlpPolicy",
        env1=env1,
        env2=env2,
        tensorboard_log="/Users/hcs/UR/hepo/cartpole_tb_logs/",
    )

    model.learn(total_timesteps=10_000)
