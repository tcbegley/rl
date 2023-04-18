import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from datasets import load_dataset
from model import RLHF
from shared import (
    create_infinite_dataloader,
    create_lr_scheduler,
    init_model,
    load_checkpoint,
    setup,
)
from tensordict.nn import (
    TensorDictModule,
    TensorDictSequential,
    ProbabilisticTensorDictModule,
)
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.prototype import tensorclass
from torch import nn
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import init_ddp, load_and_update_config
import torch.nn.functional as F
import copy
from torchrl.modules import ProbabilisticActor, ActorValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.distributions.categorical import Categorical

from env import RLHFEnv

HERE = Path(__file__).parent


class Collate(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, batch):
        batch = torch.stack(batch, dim=0).contiguous()
        batch.batch_size = []
        if self.device.type == "cuda":
            batch = batch.pin_memory()
        return batch.to(self.device)


@tensorclass
class Data:
    prompt: torch.Tensor
    target: torch.Tensor
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class PairedDataset(Dataset):
    def __init__(self, path, block_size):
        self._memmap = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __getitem__(self, idx):
        return Data(
            prompt=torch.from_numpy(
                self._memmap[idx : idx + self.block_size].astype(np.int64)
            ),
            target=torch.from_numpy(
                self._memmap[idx + 1 : idx + self.block_size + 1].astype(np.int64)
            ),
            batch_size=[self.block_size],
        )

    def __len__(self):
        # how many sequences of length block_size + 1 can we extract from the data?
        # the valid starting points for such a sequence are those tokens that aren't in
        # the final block_size positions. so it's just the length of the overall
        # sequence minus the block_size
        return len(self._memmap) - self.block_size


def create_datasets(config):
    data_dir = HERE / "nanoGPT" / "data" / config["dataset"]
    train_data = PairedDataset(data_dir / "train.bin", block_size=config["block_size"])
    val_data = PairedDataset(data_dir / "val.bin", block_size=config["block_size"])

    return train_data, val_data


def get_dataloaders(config):
    train_data, val_data = create_datasets(config)

    train_loader = create_infinite_dataloader(
        train_data, config, Collate(config["device"])
    )
    val_loader = create_infinite_dataloader(val_data, config, Collate(config["device"]))

    return train_loader, val_loader


class ActorCritic(ActorValueOperator):
    def __init__(self, base_model):

        base_model = copy.deepcopy(base_model)
        n_embd = base_model.lm_head.in_features

        # actor network
        # extract last layer to be reused by actor
        actor_head = base_model.lm_head
        base_model.lm_head = nn.Identity()

        # critic network
        value_head = nn.Linear(n_embd, 1, bias=False)

        common = TensorDictModule(base_model, in_keys=["prompt"], out_keys=["x"])

        actor_head = TensorDictModule(actor_head, in_keys=["x"], out_keys=["logits"])
        actor_head = TensorDictSequential(
            actor_head,
            ProbabilisticTensorDictModule(
                in_keys=["logits"], out_keys=["action"], distribution_class=Categorical
            ),
        )
        value_head = TensorDictModule(
            value_head, in_keys=["x"], out_keys=["state_value"]
        )

        super().__init__(common, actor_head, value_head)

    # def forward(self, x, targets=None):
    #     x = self.model(x)

    #     # actor: choses action to take from state s_t
    #     # by returning probability of each action
    #     action_prob = F.softmax(self.action_head(x), dim=-1)

    #     # critic: evaluates being in the state s_t
    #     state_values = self.value_head(x)

    #     # return values for both actor and critic as a tuple of 2 values:
    #     # 1. a list with the probability of each action over the action space
    #     # 2. the value from state s_t
    #     return action_prob, state_values

    # REWRITE OR REMOVE?!?!
    def generate(
        self,
        idx,
        max_new_tokens,
        device,
        block_size,
        use_reference=True,
        reward_model=None,
        hard_code_reward=True,
    ):
        # idx is (B, T) array of indices in the current context
        log_probs = torch.tensor([]).to(device)
        log_probs_ref = torch.tensor([]).to(device)

        values_all = torch.zeros((idx.shape[0], max_new_tokens)).to(device)
        advantages_all = torch.zeros((idx.shape[0], max_new_tokens)).to(device)

        gamma = 1
        lam = 1

        # TODO: Critic, PPO
        for i in range(max_new_tokens):
            # crop idx to the last block_size tokens
            # block_size = 256
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities

            probs_next = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs_next, num_samples=1)  # (B, 1)

            probs_idx_next = torch.gather(probs_next, 1, idx_next)
            log_probs_idx_next = torch.log(probs_idx_next)
            log_probs = torch.cat((log_probs, log_probs_idx_next), dim=1)

            if use_reference:
                logits_ref, _ = self.model(idx_cond)
                logits_ref = logits_ref[:, -1, :]  # becomes (B, C)
                probs_ref_next = F.softmax(logits_ref, dim=-1)  # (B, C)
                probs_ref_idx_next = torch.gather(probs_ref_next, 1, idx_next)
                log_probs_ref_idx_next = torch.log(probs_ref_idx_next)
                log_probs_ref = torch.cat(
                    (log_probs_ref, log_probs_ref_idx_next), dim=1
                )

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

            if i == max_new_tokens - 1:
                states = idx[:, -max_new_tokens:]
                if hard_code_reward:
                    # simple test where reward for outputting the letter 'z' (89)
                    rewards = torch.zeros_like(states, dtype=torch.float16)
                    rewards[states == 89] = 1.0
                    rewards = torch.sum(rewards, 1, keepdim=True)
                    rewards[rewards > 1] = 1

                else:
                    if self.discrete_reward:
                        rewards = reward_model.forward_reward(torch.tensor(states))[0][
                            :, 1
                        ].unsqueeze(-1)
                    else:
                        rewards = reward_model.forward_reward(torch.tensor(states))

                for t in reversed(range(max_new_tokens)):
                    if t == max_new_tokens - 1:
                        # value at last state is 0
                        delta = rewards[:].squeeze() - values_all[:, t]
                        advantages_all[:, t] = delta
                        # returns_all[:, t] = rewards[:]
                    else:
                        # rewards can only be non-zero at the last state
                        delta = gamma * values_all[:, t + 1] - values_all[:, t]
                        advantages_all[:, t] = (
                            delta + gamma * lam * advantages_all[:, t + 1]
                        )
                        # returns_all[:, t] += gamma * returns_all[:, t + 1]
        return (
            idx,
            log_probs[:, -max_new_tokens:],
            log_probs_ref[:, -max_new_tokens:],
            rewards,
            advantages_all,
        )


def train(config):
    enc = tiktoken.get_encoding("gpt2")
    # TODO: clean up...train should do just the training.
    # model creation, data loading etc. should be performed outside
    # plus align all script to have same structure and order of calls

    # model init: Actor Critic
    # FIXME: Don't like this. include it into model
    model_base, _ = init_model(config)
    a2c_model = ActorCritic(model_base)
    actor = a2c_model.get_policy_operator()
    critic = a2c_model.get_value_operator()

    # model init: Reward
    reward_model = RLHF(model_base, "reward", discrete_reward=config["discrete_reward"])
    reward_model = TensorDictModule(
        reward_model, in_keys=["input"], out_keys=["reward"]
    )

    # MODEL TO DEVICE
    reward_model.to(config["device"])
    a2c_model.to(config["device"])

    # rl_model.to(config["device"])

    # critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)

    adv_fn = GAE(value_network=critic, gamma=0.99, lmbda=0.95, average_gae=True)
    loss_fn = ClipPPOLoss(actor, critic, gamma=0.99)

    optimizer = torch.optim.AdamW(loss_fn.parameters(), lr=1e-3)

    train_loader, val_loader = get_dataloaders(config)
    last_time = time.time()
    rews_all = []
    max_iters = 100_000

    t0 = time.time()
    env = RLHFEnv(reward_model=reward_model, config=config, dataloader=train_loader)

    def get_action(td):
        prompt = torch.cat((td["prompt"], td["generated"]), dim=-1)[
            :, -config["block_size"] :
        ]
        _, _, td["action"] = actor(prompt)
        return td

    def get_values(td):
        prompt = torch.cat((td["prompt"], td["generated"]), dim=-1)[
            :, -config["block_size"] :
        ]


    for i in range(max_iters):
        td = env.rollout(
            config["episode_length"], policy=get_action, return_contiguous=False
        )
        adv_fn(td)

        loss_vals = loss_fn(td)
        loss_val = sum(
            value for key, value in loss_vals.items() if key.startswith("loss")
        )
        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Iteration {i}: {loss_val=}")


PPO_CONFIG = {
    # cardinality of the sub-samples gathered from the current data in the inner loop
    "sub_batch_size": 64,
    # optimisation steps per batch of data collected
    "num_epochs": 10,
    # clip value for PPO loss: see the equation in the intro for more context.
    "clip_epsilon": (0.2),
    "gamma": 0.99,
    "lmbda": 0.95,
    "entropy_eps": 1e-4,
}


def main():
    config = load_and_update_config("config/train_rl.yaml")
    config.update(init_ddp(config["backend"], config["device"]))
    config["ppo"] = PPO_CONFIG

    ctx = setup(config)
    train(config)

    if config["is_ddp"]:
        destroy_process_group()


if __name__ == "__main__":
    main()
