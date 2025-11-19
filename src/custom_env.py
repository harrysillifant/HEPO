import gymnasium as gym
from typing import Callable, Optional, Dict, Any

TaskPredicate = Callable[[Any, float, bool, bool, Dict[str, Any]], bool]


class RewardSplitWrapper(gym.Wrapper):
    """
    Returns the original scalar reward (unchanged) to keep compat with agents.
    Adds 'reward_components' into the info dict:
        info['reward_components'] = {'heuristic': float, 'task': float}
    Can supply task_predicate(next_obs, reward, terminated, truncated, info)
    which should return True when that step's reward should be considered the task reward.
    """

    def __init__(
        self,
        env: gym.Env,
        task_predicate: Optional[TaskPredicate] = None,
        info_key: str = "reward_components",
    ):
        super().__init__(env)
        if task_predicate is None:

            def default_pred(next_obs, reward, terminated, truncated, info):
                return bool(terminated and (not truncated) and (reward > 0))

            task_predicate = default_pred

        self.task_predicate = task_predicate
        self.info_key = info_key
        self.episode_heuristic = 0.0
        self.episode_task = 0.0

    def reset(self, **kwargs):
        self.episode_heuristic = 0.0
        self.episode_task = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        is_task = False
        try:
            is_task = bool(
                self.task_predicate(
                    next_obs, reward, terminated, truncated, info)
            )
        except Exception:
            is_task = bool(terminated and (not truncated) and (reward > 0))

        if is_task:
            task_r = float(reward)
            heuristic_r = 0.0
        else:
            task_r = 0.0
            heuristic_r = float(reward)

        info = dict(info)  # copy to avoid mutating env internals
        info[self.info_key] = {"heuristic": heuristic_r, "task": task_r}

        self.episode_heuristic += heuristic_r
        self.episode_task += task_r

        info.setdefault("episode_rewards", {})
        info["episode_rewards"]["heuristic_total"] = self.episode_heuristic
        info["episode_rewards"]["task_total"] = self.episode_task

        return next_obs, reward, terminated, truncated, info
