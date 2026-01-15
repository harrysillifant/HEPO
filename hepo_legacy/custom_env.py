from typing import Callable, Optional, Dict, Any
import numpy as np

from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv

TaskPredicate = Callable[[Any, float, bool, bool, Dict[str, Any]], bool]


class VectorizedRewardSplitWrapper(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        task_predicate: Optional[TaskPredicate] = None,
        info_key: str = "reward_components",
    ):
        super().__init__(venv)
        if task_predicate is None:

            def default_pred(next_obs, reward, terminated, truncated, info):
                return bool(
                    terminated and (not truncated)
                )  # should remove reward > 0 condition as some task rewards can be negative

            task_predicate = default_pred

        self.task_predicate = task_predicate
        self.info_key = info_key
        self.episode_heuristic = np.zeros(self.num_envs)
        self.episode_task = np.zeros(self.num_envs)

    def reset(self):
        self.episode_task = np.zeros(self.num_envs)
        self.episode_heuristic = np.zeros(self.num_envs)
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        rewards = np.asarray(rewards)

        infos = list(infos)

        for i in range(self.num_envs):
            info = dict(infos[i]) if (infos[i] is not None) else {}
            next_obs = obs[i]
            reward = float(rewards[i])

            terminated = bool(dones[i])
            truncated = False

            if isinstance(infos[i], dict):
                if "terminated" in infos[i]:
                    terminated = bool(infos[i].get("terminated", terminated))
                if "truncated" in infos[i]:
                    truncated = bool(infos[i].get("truncated", truncated))
                if infos[i].get("TimeLimit.truncated", False):
                    truncated = True

            is_task = bool(
                self.task_predicate(
                    next_obs, reward, terminated, truncated, info)
            )

            if is_task:
                task_r = float(reward)
                heuristic_r = 0.0
            else:
                task_r = 0.0
                heuristic_r = float(reward)

            info[self.info_key] = {"heuristic": heuristic_r, "task": task_r}

            self.episode_heuristic[i] += heuristic_r
            self.episode_task[i] += task_r

            info.setdefault("episode_rewards", {})
            info["episode_rewards"]["heuristic_total"] = float(
                self.episode_heuristic[i]
            )
            info["episode_rewards"]["task_total"] = float(self.episode_task[i])

            infos[i] = info

            if dones[i]:
                self.episode_heuristic[i] = 0.0
                self.episode_task[i] = 0.0

        return obs, rewards, dones, infos
