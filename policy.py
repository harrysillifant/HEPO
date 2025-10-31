import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

# 1) custom policy that contains an auxiliary head


class AuxActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, aux_hidden_dim: int = 64, **kwargs):
        # call parent: it will build feature extractor / mlp_extractor etc.
        super().__init__(*args, **kwargs)

        # features_dim is created by parent during setup
        # create a small MLP for the auxiliary objective
        self.aux_net = nn.Sequential(
            nn.Linear(self.features_dim, aux_hidden_dim),
            nn.ReLU(),
            # adjust output shape to your aux target
            nn.Linear(aux_hidden_dim, 1),
        )

    def get_aux(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Return the auxiliary prediction for the given observations.
        obs shape: (batch, *obs_shape) in SB3 convention
        """
        # extract features using the policy's extractor (works for SB3 policies)
        features = self.extract_features(obs)
        return self.aux_net(features).squeeze(-1)  # shape (batch,)


# 2) subclass PPO and override train() to include aux loss


class PPOWithAux(PPO):
    def __init__(self, *args, aux_coef: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_coef = aux_coef

    def train(self) -> None:
        # mostly the same as SB3 PPO.train(); we compute an extra aux_loss and add it
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # bookkeeping for logging (optional)
        entropy_losses, pg_losses, value_losses, aux_losses = [], [], [], []
        clip_fractions = []

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, __import__("gym").spaces.Discrete):
                    actions = actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # main eval (actor / critic)
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                clip_fractions.append(
                    torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                )

                # value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # entropy
                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # ---- YOUR AUXILIARY LOSS ----
                # compute your auxiliary prediction from the policy and an aux target.
                # You must decide where aux_target comes from (rollout_buffer, obs, next_obs, etc.).
                # Example placeholder: predict a scalar target stored in rollout_data (you'll likely need to extend the buffer)
                # Replace the next line with your actual aux target construction:
                # aux_target = rollout_data.aux_target  # <-- depends on how you store it
                #
                # For demo, let's assume target is zeros of correct shape:
                aux_pred = self.policy.get_aux(rollout_data.observations)  # (batch,)
                # <-- replace with your real target
                aux_target = torch.zeros_like(aux_pred)
                aux_loss = F.mse_loss(aux_pred, aux_target)
                aux_losses.append(aux_loss.item())

                # total loss: actor + entropy + value + aux
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.aux_coef * aux_loss
                )

                # optimization step (same as SB3)
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

                approx_kl_divs.append(
                    torch.mean(rollout_data.old_log_prob - log_prob)
                    .detach()
                    .cpu()
                    .numpy()
                )

            if (
                self.target_kl is not None
                and np.mean(approx_kl_divs) > 1.5 * self.target_kl
            ):
                break

        # log some custom metrics
        # (SB3 logger functions can be used here, e.g. logger.record)
        # e.g. logger.record("train/aux_loss", np.mean(aux_losses))
