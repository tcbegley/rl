import copy
from copy import deepcopy

import torch.nn as nn
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.distributions.categorical import Categorical

from torchrl.modules import (
    ActorValueOperator,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
)

from .transformer import init_transformer

__all__ = ["ActorCritic", "init_actor_critic"]


def penalise_repetitions(input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    score = torch.gather(scores, 2, input_ids.unsqueeze(1))
    penalty = 1.5
    # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
    score = torch.where(score < 0, score * penalty, score / penalty)
    out_scores = scores.scatter(2, input_ids.unsqueeze(1), score)
    return out_scores.to(input_ids.device)


class ActorCritic(ActorValueOperator):
    def __init__(self, base_model):
        base_model = copy.deepcopy(base_model)
        n_embd = base_model.lm_head.in_features

        # actor network
        # extract last layer to be reused by actor
        actor_head = deepcopy(base_model.lm_head)
        base_model.lm_head = nn.Identity()

        # TODO: compile base_model here?

        # critic network
        value_head = nn.Linear(n_embd, 1, bias=False)

        common = TensorDictModule(base_model, in_keys=["prompt"], out_keys=["x"])
        actor_head = TensorDictModule(actor_head, in_keys=["x"], out_keys=["raw_logits"])
        logit_scaler = TensorDictModule(penalise_repetitions, in_keys=["prompt", "raw_logits"], out_keys=["logits"])
        actor_head = SafeProbabilisticTensorDictSequential(
            actor_head,
            logit_scaler,
            SafeProbabilisticModule(
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=Categorical,
                return_log_prob=True,
            ),
        )
        value_head = TensorDictSequential(
            TensorDictModule(value_head, in_keys=["x"], out_keys=["state_value"]),
            TensorDictModule(
                lambda x: x[:, -1, :], in_keys=["state_value"], out_keys=["state_value"]
            ),
        )

        super().__init__(common, actor_head, value_head)


def init_actor_critic(config):
    model_base, _ = init_transformer(
        config, as_tensordictmodule=False, skip_compilation=True
    )
    a2c_model = ActorCritic(model_base)
    # freeze common / gpt-2 model
    a2c_model.module[0].requires_grad_(False)
    a2c_model.module[0].eval()
    a2c_model.to(config["device"])
    actor = a2c_model.get_policy_operator()
    critic = a2c_model.get_value_operator()
    critic_head = a2c_model.get_value_head()

    # FIXME: we are missing compile...
    # but we would compile TDModule...check performance issues
    return actor, critic, critic_head
