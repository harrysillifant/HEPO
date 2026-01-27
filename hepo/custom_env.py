from typing import Callable, Optional, Dict, Any
import numpy as np

from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv


class VectorizedRewardSplitWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, task_predicate: Callable):
        super().__init__(venv)
        self.task_predicate = task_predicate

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        """
        Puts task and heuristic rewards into info dict
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        for i in range(self.num_envs):
            _obs, reward, done, info = obs[i], rewards[i], dones[i], infos[i]
            is_task = self.task_predicate(_obs, reward, done, info)
            if is_task:
                info["task_reward"] = float(reward)
                info["heuristic_reward"] = 0
            else:
                info["task_reward"] = 0
                info["heuristic_reward"] = float(reward)

        return obs, rewards, dones, infos
