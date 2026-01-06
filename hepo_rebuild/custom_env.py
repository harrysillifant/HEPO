from typing import Callable, Optional, Dict, Any
import numpy as np

from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv


# class VectorizedRewardSplitWrapper(VecEnvWrapper):
#     def __init__(
#         self,
#         venv: VecEnv,
#         task_predicate: Optional[TaskPredicate] = None,
#         info_key: str = "reward_components",
#     ):
#         pass
