from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces
import torch
from typing import Union
import numpy as np


class HEPOBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        self.task_rewards = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32)
        self.heuristic_rewards = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        super().reset()

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_values_task: torch.Tensor,
        last_values_heuristic: torch.Tensor,
        dones: np.ndarray,
    ) -> None:
        super().compute_returns_and_advantages(last_values, dones)
        # Should do this on both the heuristic and task rewards
        # Convert to numpy
        last_values_task = last_values_task.clone().cpu(
        ).numpy().flatten()  # type: ignore[assignment]
        last_values_heuristic = last_values_heuristic.clone().cpu().numpy().flatten()

        last_gae_lam_task = 0
        last_gae_lam_heuristic = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values_task = last_values_task
                next_values_heuristic = last_values_heuristic

            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values_task = self.values[step + 1]
                next_values_heuristic = self.values[step + 1]
            delta_task = (
                self.rewards[step]
                + self.gamma * next_values_task * next_non_terminal
                - self.values[step]
            )
            delta_heuristic = (
                self.rewards[step]
                + self.gamma * next_values_heuristic * next_non_terminal
                - self.values[step]
            )
            last_gae_lam_task = (
                delta_task
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam_task
            )
            last_gae_lam_heuristic = (
                delta_heuristic
                + self.gamma
                * self.gae_lambda
                * next_non_terminal
                * last_gae_lam_heuristic
            )
            self.task_advantages[step] = last_gae_lam_task
            self.heuristic_advantages[step] = last_gae_lam_heuristic
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.task_returns = self.task_advantages + self.task_values
        self.heuristic_returns = self.heuristic_advantages + self.heuristic_values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        task_rewards: np.ndarray,
        heuristic_rewards: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        super().add(
            obs=obs,
            action=action,
            reward=reward,
            episode_start=episode_start,
            value=value,
            log_prob=log_prob,
        )
        self.task_rewards[self.pos] = np.array(task_rewards)
        self.heuristic_returns[self.pos] = np.array(heuristic_rewards)

    def get():
        raise NotImplementedError

    def _get_samples():
        raise NotImplementedError
