import torch
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor


from buffers import HEPOBuffer
from policies import HEPOActorCriticPolicy


class HEPOAlgorithm(OnPolicyAlgorithm):
    policy_aliases = {"MlpPolicy": HEPOActorCriticPolicy}

    def _setup_model(self):
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = HEPOBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        self.policy = HEPOActorCriticPolicy(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )

        self.policy = self.policy.to(self.device)

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
            self.policy.reset_noise(env.n_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.n_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                # type: ignore[arg-type]
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, value_outputs, log_probs = self.policy(obs_tensor)
                # task_values = values[:, 0].clone()
                # heuristic_values = values[:, 1].clone()
                # values = torch.sum(values, dim=1)
                task_values = value_outputs[:, 0]
                heuristic_values = value_outputs[:, 1]
                values = task_values + heuristic_values

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

            # shaping and task rewards are in here
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            task_rewards = np.zeros(self.n_envs)
            heuristic_rewards = np.zeros(self.n_envs)
            for i in range(self.n_envs):
                heuristic_rewards[i] = infos[i]["episode_rewards"]["heuristic_total"]
                task_rewards[i] = infos[i]["episode_rewards"]["task_total"]

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
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(
                            terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value.sum()

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                task_rewards,
                heuristic_rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                task_values,
                heuristic_values,
                log_probs,
            )

            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            value_preds = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device)
            )  # type: ignore[arg-type]
            # task_values = values[:, 0].clone()
            # heuristic_values = values[:, 1].clone()
            # values = torch.sum(values, dim=1)
            task_values = value_preds[:, 0]
            heuristic_values = value_preds[:, 1]
            values = task_values + heuristic_values

        rollout_buffer.compute_returns_and_advantage(
            last_values=values,
            last_values_task=task_values,
            last_values_heuristic=heuristic_values,
            dones=dones,
        )

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
