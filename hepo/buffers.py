from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces
import torch
from typing import Union, NamedTuple
from collections.abc import Generator
import numpy as np

from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class HEPOBuffer(RolloutBuffer):
    def reset(self):
        self.rewards_task = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32)

        self.rewards_heuristic = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.returns_task = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns_heuristic = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.values_task = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32)
        self.values_heuristic = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.advantages_task = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.advantages_heuristic = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        super().reset()

    def compute_returns_and_advantage(
        self, last_values_task, last_values_heuristic, dones
    ):
        # Convert to numpy
        last_values_task = last_values_task.clone().cpu(
        ).numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values_task
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values_task[step + 1]
            delta = (
                self.rewards_task[step]
                + self.gamma * next_values * next_non_terminal
                - self.values_task[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages_task[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns_task = self.advantages_task + self.values_task

        # Convert to numpy
        last_values_heuristic = last_values_heuristic.clone(
        ).cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values_heuristic
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values_heuristic[step + 1]
            delta = (
                self.rewards_heuristic[step]
                + self.gamma * next_values * next_non_terminal
                - self.values_heuristic[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages_heuristic[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns_heuristic = self.advantages_heuristic + self.values_heuristic

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        reward_task: np.ndarray,
        reward_heuristic: np.ndarray,
        episode_start: np.ndarray,
        value_task: torch.Tensor,
        value_heuristic: torch.Tensor,
        log_prob: torch.Tensor,
    ):
        self.rewards_task[self.pos] = np.array(reward_task)
        self.rewards_heuristic[self.pos] = np.array(reward_heuristic)
        self.values_task[self.pos] = value_task.clone().cpu().numpy().flatten()
        self.values_heuristic[self.pos] = (
            value_heuristic.clone().cpu().numpy().flatten()
        )
        super().add(
            obs=obs,
            action=action,
            reward=reward,
            episode_start=episode_start,
            value=torch.zeros((self.n_envs,)),
            log_prob=log_prob,
        )

    def get(self, batch_size: int):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values_task",
                "values_heuristic",
                "log_probs",
                "advantages_task",
                "advantages_heuristic",
                "returns",
                "returns_task",
                "returns_heuristic",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ):
        data = (
            self.observations[batch_inds],
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values_task[batch_inds].flatten(),
            self.values_heuristic[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages_task[batch_inds].flatten(),
            self.advantages_heuristic[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.returns_task[batch_inds].flatten(),
            self.returns_heuristic[batch_inds].flatten(),
        )
        return HEPOBufferSamples(*tuple(map(self.to_torch, data)))


class HEPOBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values_task: torch.Tensor
    old_values_heuristic: torch.Tensor
    old_log_prob: torch.Tensor
    advantages_task: torch.Tensor
    advantages_heuristic: torch.Tensor
    returns: torch.Tensor
    returns_task: torch.Tensor
    returns_heuristic: torch.Tensor
