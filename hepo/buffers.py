from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces
import torch
from typing import Union, NamedTuple
from collections.abc import Generator
import numpy as np

from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


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

        self.task_returns = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32)
        self.heuristic_returns = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

        self.task_values = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32)
        self.heuristic_values = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

        self.task_advantages = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.heuristic_advantages = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

        super().reset()

    def compute_returns_and_advantage(
        self,
        last_values: torch.Tensor,
        # last_values_task: torch.Tensor,
        # last_values_heuristic: torch.Tensor,
        dones: np.ndarray,
    ) -> None:
        super().compute_returns_and_advantage(last_values, dones)
        # # Convert to numpy
        # last_values_task = last_values_task.clone().cpu(
        # ).numpy().flatten()  # type: ignore[assignment]
        # last_values_heuristic = last_values_heuristic.clone().cpu().numpy().flatten()
        #
        # last_gae_lam_task = 0
        # last_gae_lam_heuristic = 0
        # for step in reversed(range(self.buffer_size)):
        #     if step == self.buffer_size - 1:
        #         next_non_terminal = 1.0 - dones.astype(np.float32)
        #         next_values_task = last_values_task
        #         next_values_heuristic = last_values_heuristic
        #     else:
        #         next_non_terminal = 1.0 - self.episode_starts[step + 1]
        #         next_values_task = self.task_values[step + 1]
        #         next_values_heuristic = self.heuristic_values[step + 1]
        #
        #     delta_task = (
        #         self.task_rewards[step]
        #         + self.gamma * next_values_task * next_non_terminal
        #         - self.task_values[step]
        #     )
        #     delta_heuristic = (
        #         self.heuristic_rewards[step]
        #         + self.gamma * next_values_heuristic * next_non_terminal
        #         - self.heuristic_values[step]
        #     )
        #     last_gae_lam_task = (
        #         delta_task
        #         + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam_task
        #     )
        #     last_gae_lam_heuristic = (
        #         delta_heuristic
        #         + self.gamma
        #         * self.gae_lambda
        #         * next_non_terminal
        #         * last_gae_lam_heuristic
        #     )
        #     self.task_advantages[step] = last_gae_lam_task
        #     self.heuristic_advantages[step] = last_gae_lam_heuristic
        # # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        # self.task_returns = self.task_advantages + self.task_values
        # self.heuristic_returns = self.heuristic_advantages + self.heuristic_values
        #

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        task_reward: np.ndarray,
        heuristic_reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        # task_value: torch.Tensor,
        # heuristic_value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        self.task_rewards[self.pos] = np.array(task_reward)
        self.heuristic_rewards[self.pos] = np.array(heuristic_reward)
        # self.task_values[self.pos] = np.array(task_value)
        # self.heuristic_values[self.pos] = np.array(heuristic_value)

        super().add(
            obs=obs,
            action=action,
            reward=reward,
            episode_start=episode_start,
            value=value,
            log_prob=log_prob,
        )

    def get(
        self, batch_size: int | None = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                # "task_values",
                # "heuristic_values",
                "log_probs",
                "advantages",
                # "task_advantages",
                # "heuristic_advantages",
                "returns",
                # "task_returns",
                # "heuristic_returns",
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
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            # self.task_values[batch_inds].flatten(),
            # self.heuristic_values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            # self.task_advantages[batch_inds].flatten(),
            # self.heuristic_advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            # self.task_returns[batch_inds].flatten(),
            # self.heuristic_returns[batch_inds].flatten(),
        )
        return HEPOBufferSamples(*tuple(map(self.to_torch, data)))


class HEPOBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    # old_task_values: torch.Tensor
    # old_heuristic_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    # task_advantages: torch.Tensor
    # heuristic_advantages: torch.Tensor
    returns: torch.Tensor
    # task_returns: torch.Tensor
    # heuristic_returns: torch.Tensor
