from typing import Callable, Optional, Dict, Any
import numpy as np

from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv


class VectorizedRewardSplitWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, task_predicate: Callable):
        super().__init__(venv)
        self.task_predicate = task_predicate
        self.heuristic_rewards = np.zeros(self.num_envs)
        self.task_rewards = np.zeros(self.num_envs)

    def reset(self):
        obs = self.venv.reset()
        self.heuristic_rewards = np.zeros(self.num_envs)
        self.task_rewards = np.zeros(self.num_envs)
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for i in range(self.num_envs):
            _obs, reward, done, info = obs[i], rewards[i], dones[i], infos[i]
            # is_task = self.task_predicate(_obs)
            breakpoint()

        # Figure out if obs is heuristic or task
        # is_task = self.task_predicate(obs)
        # Put task and heuristic rewards into infos ?dict?
        #
        return obs, rewards, dones, infos
