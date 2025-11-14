import torch
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm


class HEPO:
    def __init__(self, policy, env1, env2):
        self.pi = HEPOPolicy(policy=policy, env=env1)
        self.pi_ref = HEPOPolicy(policy=policy, env=env2)

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
        self.pi_ref.collect_rollouts(
            self.pi_ref,
            callback2,
            self.pi_ref.rollout_buffer,
            n_rollout_steps=self.pi_ref.n_steps,
        )

    def train_pi(self):
        # Train pi on a collection
        self.pi.policy.set_training_mode(True)

        pi_entropy_losses = []
        pi_pg_losses, pi_value_losses = [], []
        pi_clip_fractions = []
        continue_training = True
        batch_size = 8
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Compute \hat{J}^{\pi^i}_{HEPO}(\pi)
            for pi_rollout_data in self.pi.rollout_buffer.get(batch_size):
                pi_actions = pi_rollout_data.actions

                pi_values, pi_log_prob, pi_entropy = self.pi.policy.evaluate_actions(
                    pi_rollout_data.observations, pi_actions
                )
                pi_values = pi_values.flatten()

                pi_advantages = (
                    pi_rollout_data.advantages
                )  # advantages of pi on task rewards

                if self.normalize_advantage and len(advantages) > 1:
                    pi_advantages = (pi_advantages - pi_advantages.mean()) / (
                        pi_advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                pi_ratio = th.exp(pi_log_prob - pi_rollout_data.old_log_prob)

                pi_policy_loss_1 = pi_advantages * pi_ratio
                pi_policy_loss_2 = pi_advantages * th.clamp(
                    pi_ratio, 1 - clip_range, 1 + clip_range
                )
                pi_policy_loss = - \
                    th.min(pi_policy_loss_1, pi_policy_loss_2).mean()

                pi_pg_losses.append(pi_policy_loss.item())
                pi_clip_fraction = th.mean(
                    (th.abs(pi_ratio - 1) > pi_clip_range).float()
                ).item()
                pi_clip_fractions.append(pi_clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(
                    rollout_data.returns, values_pred
                )  # compute new value loss here
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )  # compute new loss here

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) -
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

                self.pi.policy.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.pi.policy.parameters(), self.max_grad_norm
                )
                self.pi.policy.optimizer.step()
            # Compute \hat{J}^{\pi_H^i}_{HEPO}(\pi)
            for _ in self.pi_ref.rollout_data.get(batch_size):

            self._n_updates += 1
            if not continue_training:
                break

    def train_pi_ref(self):
        pass

    def learn(self):
        """
        Train both policies,
        """
        # total_timesteps, callback = self._setup_learn(
        #     total_timesteps,
        #     callback,
        #     reset_num_timesteps,
        #     tb_log_name,
        #     progress_bar,
        # )

        assert self.env1 is not None and self.env2 is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts()

            if not continue_training:
                break

            iteration += 1

            # self._update_current

            if log_interval is not None and iteration % log_interval == 0:
                # assert self.ep_info_buffer is not None
                self.dump_logs(iteration)

            self.train_pi()
            self.train_pi_ref()

        # callback.on_training_end()

        return self


class HEPOPolicy(OnPolicyAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def collect_rollouts(self):
        pass
