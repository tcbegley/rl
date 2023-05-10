from pathlib import Path
from typing import Optional

import tiktoken
import torch

from data.shakespeare import get_dataloaders
from models.reward import init_reward_model
from models.transformer import DEFAULT_VOCAB_SIZE
from tensordict.tensordict import TensorDict
from transformers import pipeline

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase
from torchrl.envs.utils import step_mdp
from utils import load_and_update_config

HERE = Path(__file__).parent


def get_reward_factory(config):
    sentiment_pipeline = pipeline(
        model="finiteautomata/bertweet-base-sentiment-analysis"
    )

    def get_reward(text, mode):
        sent = sentiment_pipeline(text)
        if mode == '+':
            labels = torch.tensor(
                [a['label']=='POS' for a in sent],dtype=torch.float16
            ).unsqueeze(-1).to(config["device"])
        elif mode == '-':
            labels = torch.tensor(
                [a['label']=='NEG' for a in sent],dtype=torch.float16
            ).unsqueeze(-1).to(config["device"])
        else:
            raise ValueError('Unknown Mode')

        weights = torch.tensor(
            [a['score'] for a in sent],dtype=torch.float32
        ).unsqueeze(-1).to(config["device"])

        rewards = labels * weights # (B, 1)

        return rewards

    return get_reward



@torch.no_grad()
def _step(self, tensordict):
    self.step_num += 1
    prompt = tensordict["prompt"]

    # perform the action
    action = tensordict["action"].squeeze(-1)

    # compute the reward
    if self.step_num >= self.config["episode_length"]:
        # reward = self.reward_model(prompt).unsqueeze(-1)
        reward = self.get_reward([self.enc.decode(s.tolist()) for s in prompt], mode="+")
        done = torch.ones_like(reward, dtype=torch.bool)
    else:
        reward = torch.zeros((*tensordict.batch_size, 1))
        done = torch.zeros_like(reward, dtype=torch.bool)

    # The output must be written in a ``"next"`` entry
    next_prompt = torch.hstack((prompt, action[..., None]))[
        :, -self.config["block_size"] :
    ]
    out = TensorDict(
        {"next": {"prompt": next_prompt, "reward": reward, "done": done}},
        tensordict.shape,
    )
    return out


@torch.no_grad()
def _reset(self, tensordict):
    self.step_num = 0
    if tensordict is None or tensordict.is_empty():
        # if no tensordict is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input tensordict contains all the relevant
        # parameters to get started.
        tensordict = TensorDict({}, batch_size=self.config["batch_size"])

    batch = next(self.dataloader)

    out = TensorDict(
        {
            "prompt": batch.prompt[:, -self.config["block_size"] :],
            "done": torch.zeros((*batch.prompt.shape[:-1], 1, 1), dtype=torch.bool),
        },
        tensordict.shape,
    )
    return out


def _make_spec(self):
    # Under the hood, this will populate self.output_spec["observation"]
    self.observation_spec = CompositeSpec(
        prompt=BoundedTensorSpec(
            minimum=0,
            maximum=DEFAULT_VOCAB_SIZE,
            shape=(self.config["batch_size"], self.config["block_size"]),
            dtype=torch.int64,
        ),
        shape=(self.config["batch_size"],),
    )
    # since the environment is stateless, we expect the previous output as input
    self.input_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec, but the convenient
    # self.action_spec = spec is supported
    self.action_spec = BoundedTensorSpec(
        minimum=0,
        maximum=DEFAULT_VOCAB_SIZE,
        shape=(self.config["batch_size"], 1),
        dtype=torch.int64,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(
        shape=(self.config["batch_size"], 1, 1)
    )
    self.done_spec = BoundedTensorSpec(
        minimum=0,
        maximum=1,
        shape=(self.config["batch_size"], 1, 1),
        dtype=torch.bool,
    )


def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng


class RLHFEnv(EnvBase):
    batch_locked = False

    def __init__(self, get_reward=None, config=None, dataloader=None, seed=None):
        # if td_params is None:
        #     td_params = self.gen_params()
        super().__init__(device=config["device"], batch_size=[config["batch_size"]])

        # self.reward_model = reward_model
        self.get_reward = get_reward
        self.config = config
        self.dataloader = dataloader
        self.enc = tiktoken.get_encoding("gpt2")
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.step_num = 0
        # self.set_seed(seed)

    # Helpers: _make_step and gen_params
    # gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = _step
    _set_seed = _set_seed


def main():
    from torchrl.envs import check_env_specs

    config = load_and_update_config("config/train_rlhf.yaml")
    # reward_model, _ = init_reward_model(config)
    train_loader, _ = get_dataloaders(config)
    env = RLHFEnv(
        get_reward=get_reward_factory(config), dataloader=train_loader, config=config
    )

    check_env_specs(env)

    td = env.reset()

    def get_action(td):
        td["action"] = torch.randint(1, 1000, td.shape, device=config["device"])
        return td

    # rollout
    env.rollout(3, get_action, return_contiguous=False)

    print(td.shape)
    print(td)
    print(env.batch_size)
    # manual rollout
    for i in range(3):
        # td = get_action(td)
        td["action"] = torch.randint(1, 1000, td.shape)
        td = env.step(td)
        print("random step tensordict", i, td["action"], td["next", "prompt"])
        td = step_mdp(td)
    print(list(td.keys(True)))


if __name__ == "__main__":
    main()
