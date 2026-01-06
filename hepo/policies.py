import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule


class HEPOActorCriticPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # maybe need to rewrite whole _build function


MlpPolicy = HEPOActorCriticPolicy
