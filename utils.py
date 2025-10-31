from stable_baselines3.common.callbacks import BaseCallback


class HEPO(OnPolicyAlgorithm):
    def __init__(self, env, policy, heuristic_policy, **kwargs):
        super().__init__(policy, env, **kwargs)
        self.policy_H = heuristic_policy
        self.rollout_buffer_H = RolloutBuffer(
            self.n_steps,
            self.env.observation_space,
            self.env.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    def _setup_model(self) -> None:
        pass

    def compute_heuristic_reward(self, obs, action, next_obs, info):
        # Here will define the heuristic reward function given we take action at state obs
        pass

    def train(self) -> None:
        pass

    def collect_rollouts(
        self, env, callback, rollout_buffer, rollout_buffer_H, n_rollouts
    ):
        self.policy.set_training_modes(False)
        self.policy_H.set_training_modes(False)
        self.rollout_buffer.reset()
        self.rollout_buffer_H.reset()
        # Rollout B/2 trajectories for pi
        obs = env.reset()
        for i in range(self.n_rollouts):
            with torch.no_grad():
                actions, values, log_probs = self.policy.forward(obs)

            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)
            self.rollout_buffer.add(obs, actions, rewards, new_obs, dones)

        # Rollout B/2 trajectories for pi_ref
        obs = env.reset()
        for i in range(self.n_rollouts):
            with torch.no_grad():
                actions, values, log_probs = self.policy_H.forward(obs)
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            self.rollout_buffer_H.add(obs, actions, rewards, new_obs, dones)
