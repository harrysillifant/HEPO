import torch
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from gymnasium import spaces


class HEPO:
    def __init__(self, policy, env1, env2):
        self.pi = HEPOPolicy(policy=policy, env=env1)
        self.pi_H = HEPOPolicy(policy=policy, env=env2)

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
        pi_value_task_losses = []
        pi_value_heuristic_losses = []
        pi_value_task__losses = []
        pi_value_heuristic_losses = []
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

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )

                values = values.flatten()

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
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

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

                loss = pi_policy_loss + pi_H_policy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
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
            self.train_pi_H()

        # callback.on_training_end()

        return self


class HEPOPolicy(OnPolicyAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                # type: ignore[arg-type]
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(
                        clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(
                clipped_actions
            )  # shaping and task rewards are in here

            heuristic_reward = infos["episode_rewards"]["heuristic_total"]
            task_reward = infos["episode_rewards"]["task_total"]

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(
                            terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                task_reward,
                heuristic_reward,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(
                new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
