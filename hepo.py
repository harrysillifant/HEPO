import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.schedules import FloatSchedule

vec_env = make_vec_env("BipedalWalker-v3", n_envs=4)

pi = PPO("MlpPolicy", vec_env)
pi_ref = PPO("MlpPolicy", vec_env)

buff1 = RolloutBuffer(
    100,
    vec_env.observation_space,
    vec_env.action_space,
    device="cpu",
    gamma=0.99,
    gae_lambda=0.95,
)
buff2 = RolloutBuffer()


pi = PPO("MlpPolicy", vec_env)  # trained policy
pi_ref = PPO("MlpPolicy", vec_env)  # reference policy, will remove later

N_ITERATIONS = 10
N_ROLLOUTS = 10
N_STEPS = 1000

for i in range(N_ITERATIONS):
    # Rollout B/2 trajectories for pi
    for j in range(N_ROLLOUTS):
        obs = vec_env.reset()
        for step in range(N_STEPS):  # a single trajectory
            with torch.no_grad():
                actions, values, log_probs = pi.policy(
                    obs, deterministic=False)  # ????

            actions = actions.cpu().numpy()
            next_obs, reward, done, info = vec_env.step(actions)
            buff1.add(obs, actions, reward, next_obs, done)
            obs = next_obs

    # Now buff1 contains a trajectory of pi

    # Rollout B/2 trajectories for pi_ref
    for j in range(N_ROLLOUTS):
        obs = vec_env.reset()
        for step in range(N_STEPS):  # a single trajectory
            with torch.no_grad():
                actions, values, log_probs = pi_ref.policy(
                    obs, deterministic=False)
            actions = actions.cpu().numpy()
            next_obs, reward, done, info = vec_env.step(actions)
            buff2.add(obs, actions, reward, next_obs, done)
            obs = next_obs
    # Now buff2 containts a trajectory of pi_ref

    # Train pi, need trajectories (s_0, a_0, r_0, s_1, ...) from pi and pi_ref
    # what is the gradient of pi?
    # Must compute advantages via GAE
    # Compute A_r^{pi^i} and A_h^{pi^i} with GAE

    # Compute A_r^{pi_H^i} and A_h^{pi_H^i} with GAE

    # Compute task objectives and heuristic objectives

    # Update pi and  pi_ref, need trajectories from pi and pi_ref

    # update value fns

    # update alpha, use same advantages as earlier


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

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollouts):
        self.policy.set_training_modes(False)
        self.policy_H.set_training_modes(False)
        obs = torch.as_tensor(env.reset()[0]).to(self.device)
        rollout_buffer.reset()
        self.rollout_buffer_H.reset()
        # Rollout B/2 trajectories for pi
        for i in range(self.n_rollouts):
            with torch.no_grad():
                actions, values, log_probs = self.policy.forward(obs)

            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)

        # Rollout B/2 trajectories for pi_ref
