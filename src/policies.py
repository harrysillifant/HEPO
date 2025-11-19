from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule


class HEPOActorCriticPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self.value
