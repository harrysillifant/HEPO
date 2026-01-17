import numpy as np
import torch
import torch.nn.functional as F
import time
from collections import deque
import warnings

from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    ConvertCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import configure_logger, FloatSchedule
from gymnasium import spaces

from algorithm import HEPOAlgorithm


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
        pass

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

        self.piH = HEPOAlgorithm(
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

        if normalize_advantage:
            assert batch_size > 1, "`batch_size` must be greater than 1"

        if self.pi.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.pi.env.num_envs * self.pi.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={
                self.pi.n_steps
            } and n_envs={self.pi.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {
                        buffer_size
                    }`,"
                    f" after every {
                        untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {
                        buffer_size % batch_size
                    }\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.pi.n_steps} and n_envs={
                        self.pi.env.num_envs
                    })"
                )

        if self.piH.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.piH.env.num_envs * self.piH.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={
                self.piH.n_steps
            } and n_envs={self.piH.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {
                        buffer_size
                    }`,"
                    f" after every {
                        untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {
                        buffer_size % batch_size
                    }\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.piH.n_steps} and n_envs={
                        self.piH.env.num_envs
                    })"
                )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self.alpha = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        self.pi._setup_model()
        self.piH._setup_model()
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "'clip_range_vf' must be positive, "
                    "pass 'None' to deactivate vf clipping"
                )
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "HEPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        iteration = 0

        total_timesteps, pi_callback = self.pi._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name + "_pi",
            progress_bar,
        )

        _, piH_callback = self.piH._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name + "_piH",
            progress_bar,
        )

        pi_callback.on_training_start(locals(), globals())
        piH_callback.on_training_start(locals(), globals())

        assert self.pi.env is not None and self.piH.env is not None

        while self.pi.num_timesteps < total_timesteps:
            pi_continue_training = self.pi.collect_rollouts(
                self.pi.env,
                pi_callback,
                self.pi.rollout_buffer,
                n_rollout_steps=self.pi.n_steps,
            )
            piH_continue_training = self.piH.collect_rollouts(
                self.piH.env,
                piH_callback,
                self.piH.rollout_buffer,
                n_rollout_steps=self.piH.n_steps,
            )

            if not pi_continue_training:
                break

            iteration += 1
            self.pi._update_current_progress_remaining(
                self.pi.num_timesteps, total_timesteps
            )
            self.piH._update_current_progress_remaining(
                self.piH.num_timesteps, total_timesteps
            )

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert (
                    self.pi.ep_info_buffer is not None
                    and self.piH.ep_info_buffer is not None
                )
                self.pi.dump_logs(iteration)
                self.piH.dump_logs(iteration)

            self.train_pi()
            self.train_piH()

        pi_callback.on_training_end()
        piH_callback.on_training_end()

        return self

    def train_pi(self) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.pi.policy.set_training_mode(True)
        # Update optimizer learning rate
        self.pi._update_learning_rate(self.pi.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(
            self.pi._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self.pi._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses_pi, pg_losses_piH = [], []
        value_losses_task, value_losses_heuristic = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data_pi, rollout_data_piH in zip(
                self.pi.rollout_buffer.get(self.batch_size),
                self.piH.rollout_buffer.get(self.batch_size),
            ):
                actions_pi = rollout_data_pi.actions
                actions_piH = rollout_data_piH.actions
                if isinstance(self.pi.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions_pi = rollout_data_pi.actions.long().flatten()
                if isinstance(self.piH.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions_piH = rollout_data_piH.actions.long().flatten()

                values_pi, log_prob_pi, entropy = self.pi.policy.evaluate_actions(
                    rollout_data_pi.observations, actions_pi
                )
                values_piH, log_prob_piH, _ = self.pi.policy.evaluate_actions(
                    rollout_data_piH.observations, actions_piH
                )
                values_task_pi, values_heuristic_pi = (
                    values_pi[:, 0].reshape(-1, 1),
                    values_pi[:, 1].reshape(-1, 1),
                )
                values_task_piH, values_heuristic_piH = (
                    values_piH[:, 0].reshape(-1, 1),
                    values_piH[:, 1].reshape(-1, 1),
                )

                values_task_pi = values_task_pi.flatten()
                values_heuristic_pi = values_heuristic_pi.flatten()
                values_task_piH = values_task_piH.flatten()
                values_heuristic_piH = values_heuristic_piH.flatten()

                # Normalize advantage
                advantages_task_pi = rollout_data_pi.advantages_task
                advantages_heuristic_pi = rollout_data_pi.advantages_heuristic
                advantages_task_piH = rollout_data_piH.advantages_task
                advantages_heuristic_piH = rollout_data_piH.advantages_heuristic

                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if (
                    self.normalize_advantage
                    and len(advantages_task_pi)
                    and len(advantages_task_piH) > 1
                ):
                    advantages_task_pi = (
                        advantages_task_pi - advantages_task_pi.mean()
                    ) / (advantages_task_pi.std() + 1e-8)
                    advantages_heuristic_pi = (
                        advantages_heuristic_pi - advantages_heuristic_pi.mean()
                    ) / (advantages_heuristic_pi.std() + 1e-8)
                    advantages_task_piH = (
                        advantages_task_piH - advantages_task_piH.mean()
                    ) / (advantages_task_piH.std() + 1e-8)
                    advantages_heuristic_piH = (
                        advantages_heuristic_piH - advantages_heuristic_piH.mean()
                    ) / (advantages_heuristic_piH.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio_pi = torch.exp(
                    log_prob_pi - rollout_data_pi.old_log_prob)
                ratio_piH = torch.exp(
                    log_prob_piH - rollout_data_piH.old_log_prob)

                # clipped surrogate loss
                # policy_loss_1 = advantages * ratio
                # policy_loss_2 = advantages * torch.clamp(
                #     ratio, 1 - clip_range, 1 + clip_range
                # )
                # policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                policy_loss_1_pi = (
                    (1 + self.alpha) * advantages_task_pi +
                    advantages_heuristic_pi
                ) * ratio_pi
                policy_loss_2_pi = (
                    (1 + self.alpha) * advantages_task_pi +
                    advantages_heuristic_pi
                ) * torch.clamp(ratio_pi, 1 - clip_range, 1 + clip_range)
                policy_loss_pi = - \
                    torch.min(policy_loss_1_pi, policy_loss_2_pi).mean()

                policy_loss_1_piH = (
                    (1 + self.alpha) * advantages_task_piH +
                    advantages_heuristic_piH
                ) * ratio_piH
                policy_loss_2_piH = (
                    (1 + self.alpha) * advantages_task_piH +
                    advantages_heuristic_piH
                ) * torch.clamp(ratio_piH, 1 - clip_range, 1 + clip_range)
                policy_loss_piH = -torch.min(
                    policy_loss_1_piH, policy_loss_2_piH
                ).mean()

                # Logging
                pg_losses_pi.append(policy_loss_pi.item())
                pg_losses_piH.append(policy_loss_piH.item())

                clip_fraction = torch.mean(
                    (torch.abs(ratio_pi - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred_task_pi = values_task_pi
                    values_pred_heuristic_pi = values_heuristic_pi
                    values_pred_task_piH = values_task_piH
                    values_pred_heuristic_piH = values_heuristic_piH
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred_task_pi = rollout_data_pi.old_values_task + torch.clamp(
                        values_task_pi - rollout_data_pi.old_values_task,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                    values_pred_heuristic_pi = (
                        rollout_data_pi.old_values_heuristic
                        + torch.clamp(
                            values_heuristic_pi - rollout_data_pi.old_values_heuristic,
                            -clip_range_vf,
                            clip_range_vf,
                        )
                    )

                    # Is this off policy difference correct?
                    values_pred_task_piH = (
                        rollout_data_piH.old_values_task
                        + torch.clamp(
                            values_task_piH - rollout_data_piH.old_values_task,
                            -clip_range_vf,
                            clip_range_vf,
                        )
                    )
                    values_pred_heuristic_piH = (
                        rollout_data_piH.old_values_heuristic
                        + torch.clamp(
                            values_heuristic_piH - rollout_data_pi.old_values_heuristic,
                            -clip_range_vf,
                            clip_range_vf,
                        )
                    )

                # Value loss using the TD(gae_lambda) target
                value_loss_task_pi = F.mse_loss(
                    rollout_data_pi.returns_task, values_pred_task_pi
                )
                value_loss_heuristic_pi = F.mse_loss(
                    rollout_data_pi.returns_heuristic, values_pred_heuristic_pi
                )
                # Again this off policy returns is sus
                value_loss_task_piH = F.mse_loss(
                    rollout_data_piH.returns_task, values_pred_task_piH
                )
                value_loss_heuristic_piH = F.mse_loss(
                    rollout_data_piH.returns_heuristic, values_pred_heuristic_piH
                )

                value_losses_task.append(value_loss_task_pi.item())
                value_losses_heuristic.append(value_loss_heuristic_pi.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob_pi)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # loss = (
                #     policy_loss
                #     + self.pi.ent_coef * entropy_loss
                #     + self.pi.vf_coef * value_loss
                # )

                loss = (
                    policy_loss_pi
                    + policy_loss_piH
                    + self.pi.ent_coef * entropy_loss
                    + self.pi.vf_coef
                    * (
                        value_loss_task_pi
                        + value_loss_heuristic_pi
                        + value_loss_task_piH
                        + value_loss_heuristic_piH
                    )
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob_pi - rollout_data_pi.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) -
                                   log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.pi.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {
                                approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.pi.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(
                    self.pi.policy.parameters(), self.pi.max_grad_norm
                )
                self.pi.policy.optimizer.step()

            self.pi._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.pi.rollout_buffer.values.flatten(),
            self.pi.rollout_buffer.returns.flatten(),
        )

        # Logs
        self.pi.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.pi.logger.record(
            "train/policy_gradient_loss", np.mean(pg_losses_pi))
        self.pi.logger.record(
            "train/off_policy_gradient_loss", np.mean(pg_losses_piH))

        self.pi.logger.record("train/value_loss_task",
                              np.mean(value_losses_task))
        self.pi.logger.record(
            "train/value_loss_heuristic", np.mean(value_losses_heuristic)
        )

        self.pi.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.pi.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.pi.logger.record("train/loss", loss.item())
        self.pi.logger.record("train/explained_variance", explained_var)
        if hasattr(self.pi.policy, "log_std"):
            self.pi.logger.record(
                "train/std", torch.exp(self.pi.policy.log_std).mean().item()
            )

        self.pi.logger.record(
            "train/n_updates", self.pi._n_updates, exclude="tensorboard"
        )
        self.pi.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.pi.logger.record("train/clip_range_vf", clip_range_vf)

    def train_piH(self) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.piH.policy.set_training_mode(True)
        # Update optimizer learning rate
        self.piH._update_learning_rate(self.piH.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(
            self.piH._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self.piH._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses_pi, pg_losses_piH = [], []
        value_losses_task, value_losses_heuristic = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data_pi, rollout_data_piH in zip(
                self.pi.rollout_buffer.get(self.batch_size),
                self.piH.rollout_buffer.get(self.batch_size),
            ):
                actions_pi = rollout_data_pi.actions
                actions_piH = rollout_data_piH.actions
                if isinstance(self.pi.action_space, spaces.Discrete):
                    actions_pi = rollout_data_pi.actions.long().flatten()
                if isinstance(self.piH.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions_piH = rollout_data_piH.actions.long().flatten()

                values_pi, log_prob_pi, _ = self.piH.policy.evaluate_actions(
                    rollout_data_pi.observations, actions_pi
                )
                values_piH, log_prob_piH, entropy = self.piH.policy.evaluate_actions(
                    rollout_data_piH.observations, actions_piH
                )

                values_task_pi, values_heuristic_pi = (
                    values_pi[:, 0].reshape(-1, 1),
                    values_pi[:, 1].reshape(-1, 1),
                )
                values_task_piH, values_heuristic_piH = (
                    values_piH[:, 0].reshape(-1, 1),
                    values_piH[:, 1].reshape(-1, 1),
                )

                values_task_pi = values_task_pi.flatten()
                values_heuristic_pi = values_heuristic_pi.flatten()
                values_task_piH = values_task_piH.flatten()
                values_heuristic_piH = values_heuristic_piH.flatten()

                # Normalize advantage
                advantages_heuristic_pi = rollout_data_pi.advantages_heuristic
                advantages_heuristic_piH = rollout_data_piH.advantages_heuristic
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if (
                    self.normalize_advantage
                    and len(advantages_heuristic_pi) > 1
                    and len(advantages_heuristic_piH) > 1
                ):
                    advantages_heuristic_pi = (
                        advantages_heuristic_pi - advantages_heuristic_pi.mean()
                    ) / (advantages_heuristic_pi.std() + 1e-8)

                    advantages_heuristic_piH = (
                        advantages_heuristic_piH - advantages_heuristic_piH.mean()
                    ) / (advantages_heuristic_piH.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio_pi = torch.exp(
                    log_prob_pi - rollout_data_pi.old_log_prob)
                ratio_piH = torch.exp(
                    log_prob_piH - rollout_data_piH.old_log_prob)

                # clipped surrogate loss
                policy_loss_1_pi = advantages_heuristic_pi * ratio_pi
                policy_loss_2_pi = advantages_heuristic_pi * torch.clamp(
                    ratio_pi, 1 - clip_range, 1 + clip_range
                )
                policy_loss_pi = - \
                    torch.min(policy_loss_1_pi, policy_loss_2_pi).mean()

                policy_loss_1_piH = advantages_heuristic_piH * ratio_piH
                policy_loss_2_piH = advantages_heuristic_piH * torch.clamp(
                    ratio_piH, 1 - clip_range, 1 + clip_range
                )
                policy_loss_piH = -torch.min(
                    policy_loss_1_piH, policy_loss_2_piH
                ).mean()

                # Logging
                pg_losses_pi.append(policy_loss_pi.item())
                pg_losses_piH.append(policy_loss_piH.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio_piH - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred_task_pi = values_task_pi
                    values_pred_heuristic_pi = values_heuristic_pi
                    values_pred_task_piH = values_task_piH
                    values_pred_heuristic_piH = values_heuristic_piH
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred_task_pi = rollout_data_pi.old_values_task + torch.clamp(
                        values_task_pi - rollout_data_pi.old_values_task,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                    values_pred_heuristic_pi = (
                        rollout_data_pi.old_values_heuristic
                        + torch.clamp(
                            values_heuristic_pi - rollout_data_pi.old_values_heuristic,
                            -clip_range_vf,
                            clip_range_vf,
                        )
                    )
                    values_pred_task_piH = (
                        rollout_data_piH.old_values_task
                        + torch.clamp(
                            values_task_piH - rollout_data_piH.old_values_task,
                            -clip_range_vf,
                            clip_range_vf,
                        )
                    )
                    values_pred_heuristic_piH = (
                        rollout_data_piH.old_values_heuristic
                        + torch.clamp(
                            values_heuristic_piH
                            - rollout_data_piH.old_values_heuristic,
                            -clip_range_vf,
                            clip_range_vf,
                        )
                    )

                # Value loss using the TD(gae_lambda) target
                value_loss_task_pi = F.mse_loss(
                    rollout_data_pi.returns_task, values_pred_task_pi
                )
                value_loss_heuristic_pi = F.mse_loss(
                    rollout_data_pi.returns_heuristic, values_pred_heuristic_pi
                )
                value_loss_task_piH = F.mse_loss(
                    rollout_data_piH.returns_task, values_pred_task_piH
                )
                value_loss_heuristic_piH = F.mse_loss(
                    rollout_data_piH.returns_heuristic, values_pred_heuristic_piH
                )

                value_losses_task.append(value_loss_task_piH.item())
                value_losses_heuristic.append(value_loss_heuristic_piH.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob_piH)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss_pi
                    + policy_loss_piH
                    + self.piH.ent_coef * entropy_loss
                    + self.piH.vf_coef
                    * (
                        value_loss_task_pi
                        + value_loss_heuristic_pi
                        + value_loss_task_piH
                        + value_loss_heuristic_piH
                    )
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob_piH - rollout_data_piH.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) -
                                   log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.piH.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {
                                approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.piH.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(
                    self.piH.policy.parameters(), self.piH.max_grad_norm
                )
                self.piH.policy.optimizer.step()

            self.piH._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.piH.rollout_buffer.values.flatten(),
            self.piH.rollout_buffer.returns.flatten(),
        )

        # Logs
        self.piH.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.piH.logger.record(
            "train/policy_gradient_loss", np.mean(pg_losses_piH))
        self.piH.logger.record(
            "train/off_policy_gradient_loss", np.mean(pg_losses_pi))
        self.piH.logger.record("train/value_loss_task",
                               np.mean(value_losses_task))
        self.piH.logger.record(
            "train/value_loss_heuristic", np.mean(value_losses_heuristic)
        )
        self.piH.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.piH.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.piH.logger.record("train/loss", loss.item())
        self.piH.logger.record("train/explained_variance", explained_var)
        if hasattr(self.piH.policy, "log_std"):
            self.piH.logger.record(
                "train/std", torch.exp(self.piH.policy.log_std).mean().item()
            )

        self.piH.logger.record(
            "train/n_updates", self.piH._n_updates, exclude="tensorboard"
        )
        self.piH.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.piH.logger.record("train/clip_range_vf", clip_range_vf)

    def update_alpha(self):
        grads = []
        alphas = []
        for epoch in range(self.n_epochs):
            for rollout_data_pi, rollout_data_piH in zip(
                self.pi.rollout_buffer.get(self.batch_size),
                self.piH.rollout_buffer.get(self.batch_size),
            ):
                advantages_task_pi = rollout_data_pi.advantages_task
                advantages_task_piH = rollout_data_piH.advantages_task

                if (
                    self.normalize_advantage
                    and len(advantages_task_pi) > 1
                    and len(advantages_task_piH) > 1
                ):
                    advantages_task_pi = (
                        advantages_task_pi - advantages_task_pi.mean()
                    ) / (advantages_task_pi.std() + 1e-8)

                    advantages_task_piH = (
                        advantages_task_piH - advantages_task_piH.mean()
                    ) / (advantages_task_piH.std() + 1e-8)

                # THESE ARE SUPPOSED TO BE OFF-POLICY ADVANTAGES

                self.alpha = self.alpha - self.lr / 2 * (
                    advantages_task_pi.mean() + advantages_task_piH.mean()
                )
