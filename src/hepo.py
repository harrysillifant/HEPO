import numpy as np
import torch
import torch.nn.functional as F
import time
from collections import deque

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import configure_logger
from gymnasium import spaces

from algorithm import HEPOAlgorithm
from policies import HEPOActorCriticPolicy


class HEPO:
    def __init__(
        self,
        policy,
        env1,
        env2,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl=None,
        stats_window_size=100,
        tensorboard_log=None,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    ):
        self.pi = HEPOAlgorithm(
            policy=policy,
            env=env1,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.pi_H = HEPOAlgorithm(
            policy=policy,
            env=env2,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self.pi._setup_model()
            self.pi_H._setup_model()

    def collect_rollouts(self, callback1=None, callback2=None):
        """
        Generate two buffers
        """
        self.pi.collect_rollouts(
            self.pi.env,
            callback1,
            self.pi.rollout_buffer,
            n_rollout_steps=self.pi.n_steps,
        )
        self.pi_H.collect_rollouts(
            self.pi_H,
            callback2,
            self.pi_H.rollout_buffer,
            n_rollout_steps=self.pi_H.n_steps,
        )

    def train_pi(self):
        # Switch to train mode (this affects batch norm / dropout)
        self.pi.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(
            self._current_progress_remaining)  # type: ignore[operator]

        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pi_pg_losses = []
        pi_H_pg_losses = []
        task_value_losses = []
        heuristic_value_losses = []
        pi_clip_fractions = []
        pi_H_clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for pi_rollout_data, pi_H_rollout_data in zip(
                self.pi.rollout_buffer.get(self.batch_size),
                self.pi_H.rollout_buffer.get(self.batch_size),
            ):
                pi_actions = pi_rollout_data.actions
                pi_H_actions = pi_H_rollout_data.actions

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    pi_actions = pi_actions.long().flatten()
                    pi_H_actions = pi_H_actions.long().flatten()

                # values, log_prob, entropy = self.policy.evaluate_actions(
                #     rollout_data.observations, actions
                # )
                (task_values, heuristic_values), pi_log_prob, pi_entropy = (
                    self.pi.policy.evaluate_actions(
                        pi_rollout_data.observations, pi_actions
                    )
                )
                _, pi_H_log_prob, pi_H_entropy = self.pi.policy.evaluate_actions(
                    pi_H_rollout_data.observation, pi_H_actions
                )

                # maybe get pi_H values here as well

                task_values = task_values.flatten()
                heuristic_values = heuristic_values.flatten()

                # Normalize advantage
                pi_task_advantages = pi_rollout_data.task_advantages
                pi_heuristic_advantages = pi_rollout_data.heuristic_advantages
                pi_H_task_advantages = pi_H_rollout_data.task_advantages
                pi_H_heuristic_advantages = pi_H_rollout_data.heuristic_advantages

                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if (
                    self.normalize_advantage
                    and len(pi_task_advantages) > 1
                    and len(pi_H_task_advantages) > 1
                ):
                    pi_task_advantages = (
                        pi_task_advantages - pi_task_advantages.mean()
                    ) / (pi_task_advantages.std() + 1e-8)
                    pi_heuristic_advantages = (
                        pi_heuristic_advantages - pi_heuristic_advantages.mean()
                    ) / (pi_heuristic_advantages.std() + 1e-8)
                    pi_H_task_advantages = (
                        pi_H_task_advantages - pi_H_task_advantages.mean()
                    ) / (pi_H_task_advantages.std() + 1e-8)
                    pi_H_heuristic_advantages = (
                        pi_H_heuristic_advantages - pi_H_heuristic_advantages.mean()
                    ) / (pi_H_heuristic_advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                # LOG PROB IS WRONG HERE
                pi_ratio = torch.exp(
                    pi_log_prob - pi_rollout_data.old_log_prob)
                pi_H_ratio = torch.exp(
                    pi_H_log_prob - pi_H_rollout_data.old_log_prob)

                # clipped surrogate loss
                # policy_loss_1 = advantages * ratio
                # policy_loss_2 = advantages * th.clamp(
                #     ratio, 1 - clip_range, 1 + clip_range
                # )
                # policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pi_U = (1 + self.alpha) * pi_task_advantages + \
                    pi_heuristic_advantages
                pi_H_U = (
                    1 + self.alpha
                ) * pi_H_task_advantages + pi_H_heuristic_advantages
                pi_policy_loss_1 = pi_U * pi_ratio
                pi_policy_loss_2 = pi_U * torch.clamp(
                    pi_ratio, 1 - clip_range, 1 + clip_range
                )
                pi_policy_loss = - \
                    torch.min(pi_policy_loss_1, pi_policy_loss_2).mean()
                pi_H_policy_loss_1 = pi_H_U * pi_H_ratio
                pi_H_policy_loss_2 = pi_H_U * torch.clamp(
                    pi_H_ratio, 1 - clip_range, 1 + clip_range
                )
                pi_H_policy_loss = - \
                    torch.min(pi_H_policy_loss_1, pi_H_policy_loss_2)

                # Logging
                pi_pg_losses.append(pi_policy_loss.item())
                pi_H_pg_losses.append(pi_H_policy_loss.item())
                pi_clip_fraction = torch.mean(
                    (torch.abs(pi_ratio - 1) > clip_range).float()
                ).item()
                pi_clip_fractions.append(pi_clip_fraction)
                pi_H_clip_fraction = torch.mean(
                    (torch.abs(pi_H_ratio - 1) > clip_range).float()
                ).item()
                pi_H_clip_fractions.append(pi_H_clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    task_values_pred = task_values
                    heuristic_values_pred = heuristic_values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    task_values_pred = pi_rollout_data.old_values + torch.clamp(
                        task_values - pi_rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                    heuristic_values_pred = pi_rollout_data.old_values + torch.clamp(
                        heuristic_values - pi_rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )

                # Value loss using the TD(gae_lambda) target
                task_value_loss = F.mse_loss(
                    pi_rollout_data.task_returns, task_values_pred
                )
                task_value_losses.append(task_value_loss.item())
                heuristic_value_loss = F.mse_loss(
                    pi_rollout_data.heuristic_returns, heuristic_values_pred
                )
                heuristic_value_losses.append(heuristic_value_loss.item())

                if pi_entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-pi_log_prob)
                else:
                    entropy_loss = -torch.mean(pi_entropy)

                entropy_losses.append(entropy_loss.item())

                # Note entropy loss only depends on the entropy calculated from the pi trajectory
                loss = (
                    pi_policy_loss
                    + pi_H_policy_loss
                    + self.vf_coef * task_value_loss
                    + self.vf_coef * heuristic_value_loss
                    + self.ent_coef * entropy_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = pi_log_prob - pi_rollout_data.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) -
                                   log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {
                                approx_kl_div:.2f}"
                        )
                    break

                self.pi.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.pi.policy.parameters(), self.max_grad_norm
                )
                self.pi.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.pi.logger.record("train/pi/entropy_loss", np.mean(entropy_losses))
        self.pi.logger.record(
            "train/pi/pi_policy_gradient_loss", np.mean(pi_pg_losses))
        self.pi.logger.record(
            "train/pi/pi_H_policy_gradient_loss", np.mean(pi_H_pg_losses)
        )
        # self.pi.logger.record("train/value_loss", np.mean(value_losses))
        self.pi.logger.record("train/pi/task_value_loss",
                              np.mean(task_value_losses))
        self.pi.logger.record("train/pi/approx_kl", np.mean(approx_kl_divs))
        self.pi.logger.record("train/pi/pi_clip_fraction",
                              np.mean(pi_clip_fractions))
        self.pi.logger.record(
            "train/pi/pi_H_clip_fractions", np.mean(pi_H_clip_fractions)
        )
        self.pi.logger.record("train/pi/loss", loss.item())
        self.pi.logger.record("train/pi/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.pi.logger.record(
                "train/pi/std", torch.exp(self.pi_policy.log_std).mean().item()
            )

        self.pi.logger.record(
            "train/pi/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.pi.logger.record("train/pi/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.pi.logger.record("train/pi/clip_range_vf", clip_range_vf)

    def train_pi_H(self):
        # Switch to train mode (this affects batch norm / dropout)
        self.pi_H.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy_H.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(
            self._current_progress_remaining)  # type: ignore[operator]

        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pi_pg_losses = []
        pi_H_pg_losses = []
        task_value_losses = []
        heuristic_value_losses = []
        pi_clip_fractions = []
        pi_H_clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for pi_rollout_data, pi_H_rollout_data in zip(
                self.pi.rollout_buffer.get(self.batch_size),
                self.pi_H.rollout_buffer.get(self.batch_size),
            ):
                pi_actions = pi_rollout_data.actions
                pi_H_actions = pi_H_rollout_data.actions

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    pi_actions = pi_actions.long().flatten()
                    pi_H_actions = pi_H_actions.long().flatten()

                # values, log_prob, entropy = self.policy.evaluate_actions(
                #     rollout_data.observations, actions
                # )
                _, pi_log_prob, pi_entropy = self.pi.policy.evaluate_actions(
                    pi_rollout_data.observations, pi_actions
                )
                (task_values, heuristic_values), pi_H_log_prob, pi_H_entropy = (
                    self.pi.policy.evaluate_actions(
                        pi_H_rollout_data.observation, pi_H_actions
                    )
                )

                # maybe get pi_H values here as well

                task_values = task_values.flatten()
                heuristic_values = heuristic_values.flatten()

                # Normalize advantage
                pi_heuristic_advantages = pi_rollout_data.heuristic_advantages
                pi_H_heuristic_advantages = pi_H_rollout_data.heuristic_advantages

                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if (
                    self.normalize_advantage
                    and len(pi_heuristic_advantages) > 1
                    and len(pi_H_heuristic_advantages) > 1
                ):
                    pi_heuristic_advantages = (
                        pi_heuristic_advantages - pi_heuristic_advantages.mean()
                    ) / (pi_heuristic_advantages.std() + 1e-8)
                    pi_H_heuristic_advantages = (
                        pi_H_heuristic_advantages - pi_H_heuristic_advantages.mean()
                    ) / (pi_H_heuristic_advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                # LOG PROB IS WRONG HERE
                pi_ratio = torch.exp(
                    pi_log_prob - pi_rollout_data.old_log_prob)
                pi_H_ratio = torch.exp(
                    pi_H_log_prob - pi_H_rollout_data.old_log_prob)

                # clipped surrogate loss
                # policy_loss_1 = advantages * ratio
                # policy_loss_2 = advantages * th.clamp(
                #     ratio, 1 - clip_range, 1 + clip_range
                # )
                # policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pi_policy_loss_1 = pi_heuristic_advantages * pi_ratio
                pi_policy_loss_2 = pi_heuristic_advantages * torch.clamp(
                    pi_ratio, 1 - clip_range, 1 + clip_range
                )
                pi_policy_loss = - \
                    torch.min(pi_policy_loss_1, pi_policy_loss_2).mean()
                pi_H_policy_loss_1 = pi_H_heuristic_advantages * pi_H_ratio
                pi_H_policy_loss_2 = pi_H_heuristic_advantages * torch.clamp(
                    pi_H_ratio, 1 - clip_range, 1 + clip_range
                )
                pi_H_policy_loss = - \
                    torch.min(pi_H_policy_loss_1, pi_H_policy_loss_2)

                # Logging
                pi_pg_losses.append(pi_policy_loss.item())
                pi_H_pg_losses.append(pi_H_policy_loss.item())
                pi_clip_fraction = torch.mean(
                    (torch.abs(pi_ratio - 1) > clip_range).float()
                ).item()
                pi_clip_fractions.append(pi_clip_fraction)
                pi_H_clip_fraction = torch.mean(
                    (torch.abs(pi_H_ratio - 1) > clip_range).float()
                ).item()
                pi_H_clip_fractions.append(pi_H_clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    task_values_pred = task_values
                    heuristic_values_pred = heuristic_values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    task_values_pred = pi_H_rollout_data.old_values + torch.clamp(
                        task_values - pi_H_rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                    heuristic_values_pred = pi_H_rollout_data.old_values + torch.clamp(
                        heuristic_values - pi_H_rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )

                # Value loss using the TD(gae_lambda) target
                task_value_loss = F.mse_loss(
                    pi_H_rollout_data.task_returns, task_values_pred
                )
                task_value_losses.append(task_value_loss.item())
                heuristic_value_loss = F.mse_loss(
                    pi_H_rollout_data.heuristic_returns, heuristic_values_pred
                )
                heuristic_value_losses.append(heuristic_value_loss.item())

                if pi_H_entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-pi_log_prob)
                else:
                    entropy_loss = -torch.mean(pi_H_entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    pi_policy_loss
                    + pi_H_policy_loss
                    + self.vf_coef * task_value_loss
                    + self.vf_coef * heuristic_value_loss
                    + self.ent_coef * entropy_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = pi_H_log_prob - pi_H_rollout_data.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) -
                                   log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {
                                approx_kl_div:.2f}"
                        )
                    break

                self.pi_H.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.pi_H.policy.parameters(), self.max_grad_norm
                )
                self.pi_H.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.pi_H.logger.record(
            "train/pi_H/entropy_loss", np.mean(entropy_losses))
        self.pi_H.logger.record(
            "train/pi_H/pi_policy_gradient_loss", np.mean(pi_pg_losses)
        )
        self.pi_H.logger.record(
            "train/pi_H/pi_H_policy_gradient_loss", np.mean(pi_H_pg_losses)
        )
        # self.pi_H.logger.record("train/value_loss", np.mean(value_losses))
        self.pi_H.logger.record(
            "train/pi_H/task_value_loss", np.mean(task_value_losses)
        )
        self.pi_H.logger.record("train/pi_H/approx_kl",
                                np.mean(approx_kl_divs))
        self.pi_H.logger.record(
            "train/pi_H/pi_clip_fraction", np.mean(pi_clip_fractions)
        )
        self.pi_H.logger.record(
            "train/pi_H/pi_H_clip_fractions", np.mean(pi_H_clip_fractions)
        )
        self.pi_H.logger.record("train/pi_H/loss", loss.item())
        self.pi_H.logger.record("train/pi_H/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.pi_H.logger.record(
                "train/pi_H/std", torch.exp(
                    self.pi_policy.log_std).mean().item()
            )

        self.pi_H.logger.record(
            "train/pi_H/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.pi_H.logger.record("train/pi_H/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.pi_H.logger.record("train/pi_H/clip_range_vf", clip_range_vf)

    def train_alpha(self):
        pass

    #
    # def _setup_learn(
    #     self,
    #     total_timesteps,
    #     callback=None,
    #     reset_num_timesteps=True,
    #     tb_log_name="run",
    #     progress_bar=False,
    # ):
    #     self.start_time = time.time_ns()
    #     if self.pi.ep_info_buffer is None or reset_num_timesteps:
    #         self.pi.ep_info_buffer = deque(maxlen=self.pi._stats_window_size)
    #         self.pi.ep_success_buffer = deque(
    #             maxlen=self.pi._stats_window_size)
    #
    #     if self.pi_H.ep_info_buffer is None or reset_num_timesteps:
    #         self.pi_H.ep_info_buffer = deque(
    #             maxlen=self.pi_H._stats_window_size)
    #         self.pi_H.ep_success_buffer = deque(
    #             maxlen=self.pi_H._stats_window_size)
    #
    #     if self.pi.action_noise is not None:
    #         self.pi.action_noise.reset()
    #
    #     if self.pi_H.action_noise is not None:
    #         self.pi_H.action_noise.reset()
    #
    #     if reset_num_timesteps:
    #         self.num_timesteps = 0
    #         self._episode_num = 0
    #     else:
    #         total_timesteps += self.num_timesteps
    #
    #     self._total_timesteps = total_timesteps
    #     self._num_timesteps_at_start = self.num_timesteps
    #
    #     if reset_num_timesteps or self.pi._last_obs is None:
    #         assert self.pi.env is not None
    #         self.pi._last_obs = self.pi.env.reset()  # type: ignore[assignment]
    #         self.pi._last_episode_starts = np.ones(
    #             (self.pi.env.num_envs,), dtype=bool)
    #         # Retrieve unnormalized observation for saving into the buffer
    #         if self.pi._vec_normalize_env is not None:
    #             self.pi._last_original_obs = (
    #                 self.pi._vec_normalize_env.get_original_obs()
    #             )
    #
    #     if reset_num_timesteps or self.pi_H._last_obs is None:
    #         assert self.pi_H.env is not None
    #         # type: ignore[assignment]
    #         self.pi_H._last_obs = self.pi_H.env.reset()
    #         self.pi_H._last_episode_starts = np.ones(
    #             (self.pi_H.env.num_envs,), dtype=bool
    #         )
    #         # Retrieve unnormalized observation for saving into the buffer
    #         if self.pi_H._vec_normalize_env is not None:
    #             self.pi_H._last_original_obs = (
    #                 self.pi_H._vec_normalize_env.get_original_obs()
    #             )
    #
    #     # Configure logger's outputs if no logger was passed
    #     if not self.pi._custom_logger:
    #         self._logger = configure_logger(
    #             self.pi.verbose,
    #             self.pi_H.tensorboard_log,
    #             tb_log_name,
    #             reset_num_timesteps,
    #         )
    #
    #     if not self.pi_H._custom_logger:
    #         self.pi_H._logger = configure_logger(
    #             self.pi_H.verbose,
    #             self.pi_H.tensorboard_log,
    #             tb_log_name,
    #             reset_num_timesteps,
    #         )
    #
    #     # Create eval callback if needed
    #     callback = self._init_callback(callback, progress_bar)
    #
    #     return total_timesteps, callback
    #
    # def _init_callback(
    #     self,
    #     callback: MaybeCallback,
    #     progress_bar: bool = False,
    # ) -> BaseCallback:
    #     pass
    #     # Parameters/functions that shoudlnt be individual to either policy
    #     # self.num_timesteps
    #     # self.update_current_progress_remaining()
    #     # self._total_timesteps
    #     #
    #     # Unsure about
    #     # self.action_noise
    #     # self._episode_n
    #     # self._init_callback
    #
    # def _update_current_progress_remaining(self, num_timesteps, total_timesteps):
    #     self._current_progress_remaining = 1.0 - float(num_timesteps) / float(
    #         total_timesteps
    #     )
    #
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Train both policies,
        """
        iteration = 0

        # total_timesteps, callback = self._setup_learn(
        #     total_timesteps,
        #     callback,
        #     reset_num_timesteps,
        #     tb_log_name,
        #     progress_bar,
        # )

        # callback.on_training_start(locals(), globals())

        assert self.pi.env is not None and self.pi_H.env is not None

        self.num_timesteps = 0
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts()

            if not continue_training:
                break

            # self._update_current_progress_remaining(
            #     self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                assert self.pi.ep_info_buffer is not None
                self.pi.dump_logs(iteration)
                assert self.pi_H.ep_info_buffer is not None
                self.pi_H.dump_logs(iteration)

            self.train_pi()
            self.train_pi_H()
            self.train_alpha()
            self.num_timesteps += 1

        # callback.on_training_end()

        return self
