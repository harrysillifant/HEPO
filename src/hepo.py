import torch
import torch.nn.functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from gymnasium import spaces

from algorithm import HEPOAlgorithm


class HEPO:
    def __init__(self, policy, env1, env2):
        self.pi = HEPOAlgorithm(policy=policy, env=env1)
        self.pi_H = HEPOAlgorithm(policy=policy, env=env2)

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
                (task_values, heuristic_values), log_prob, entropy = (
                    self.pi.policy.evaluate_actions(
                        pi_rollout_data.observations, pi_actions
                    )
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
                pi_ratio = torch.exp(log_prob - pi_rollout_data.old_log_prob)
                pi_H_ratio = torch.exp(
                    log_prob - pi_H_rollout_data.old_log_prob)

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

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # loss = (
                #     policy_loss
                #     + self.ent_coef * entropy_loss
                #     + self.vf_coef * value_loss
                # )

                loss = (
                    pi_policy_loss
                    + pi_H_policy_loss
                    + self.vf_coef * task_value_loss
                    + self.vf_coef * heuristic_value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - pi_rollout_data.old_log_prob
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

                # Optimization step
                self.pi_policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(
                    self.pi_policy.parameters(), self.max_grad_norm
                )
                self.pi_policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates",
                           self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def train_pi_H(self):
        pass

    def train_alpha(self):
        pass

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

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env1 is not None and self.env2 is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.pi.collect_rollouts()
            continue_training = self.pi_H.collect_rollouts()

            if not continue_training:
                break

            self._update_current_progress_remaining(
                self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.dump_logs(iteration)

            self.train_pi()
            self.train_pi_H()
            self.train_alpha()

        callback.on_training_end()

        return self
