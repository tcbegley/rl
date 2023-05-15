from typing import Optional

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tensordict import MemmapTensor, tensorclass
from tqdm import tqdm

from .utils import create_infinite_dataloader


class Collate(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, batch):
        if self.device.type == "cuda":
            batch = batch.pin_memory()
        return batch.to(self.device)


@tensorclass
class PairwiseDataset:
    prompt: torch.Tensor
    chosen: torch.Tensor
    rejected: torch.Tensor
    reward: Optional[torch.Tensor] = None

    @staticmethod
    def _encode(sample, max_length):
        enc = tiktoken.get_encoding("gpt2")

        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        prompt = enc.encode(
            f"<|startoftext|>{prompt}<|endoftext|>", allowed_special="all"
        )[-max_length:]
        chosen = enc.encode(
            f"<|startoftext|>{chosen}<|endoftext|>", allowed_special="all"
        )[-max_length:]
        rejected = enc.encode(
            f"<|startoftext|>{rejected}<|endoftext|>", allowed_special="all"
        )[-max_length:]
        return prompt, chosen, rejected

    @classmethod
    def from_dataset(cls, dataset, max_length):
        # we perform two passes over the dataset. during the first we determine which
        # datapoints to skip. during the second we load the unskipped examples into
        # a pre-allocated memory map. while we do end up paying the cost of iteration
        # and encoding twice, it means we are able to load the full dataset into the
        # memory map without ever having to hold the whole thing in memory
        indices_to_skip = set()
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
            if len(sample["chosen"].split()) < 5 or len(sample["rejected"].split()) < 5:
                indices_to_skip.add(idx)
                continue

            # prompt, chosen, rejected = cls._encode(sample, max_length)

            # if chosen == rejected:
            #     print("skipping because identical response")
            #     indices_to_skip.add(idx)

        n_examples = len(dataset) - len(indices_to_skip)
        data = cls(
            prompt=MemmapTensor(n_examples, max_length, dtype=torch.int32),
            chosen=MemmapTensor(n_examples, max_length, dtype=torch.int32),
            rejected=MemmapTensor(n_examples, max_length, dtype=torch.int32),
            batch_size=[len(dataset)],
        )
        i = 0

        for idx, sample in tqdm(enumerate(dataset), total=n_examples):
            if idx in indices_to_skip:
                continue

            prompt, chosen, rejected = cls._encode(sample, max_length)

            data[i] = cls(
                prompt=F.pad(
                    torch.Tensor(prompt), (max_length - len(prompt), 0), value=0
                ),
                chosen=F.pad(
                    torch.Tensor(chosen), (max_length - len(chosen), 0), value=0
                ),
                rejected=F.pad(
                    torch.Tensor(rejected), (max_length - len(rejected), 0), value=0
                ),
                batch_size=[],
            )
            i += 1

        return data


def create_datasets(config):
    # Make pairwise datasets for training
    print("Creating pairwise datasets")
    data_path = "CarperAI/openai_summarize_comparisons"
    train_data = PairwiseDataset.from_dataset(
        load_dataset(data_path, split="train"), max_length=config["block_size"]
    )
    val_data = PairwiseDataset.from_dataset(
        load_dataset(data_path, split="test"), max_length=config["block_size"]
    )

    return train_data, val_data


def get_reward_dataloaders(config):
    train_data, val_data = create_datasets(config)
    train_data.memmap_()
    val_data.memmap_()

    train_loader = create_infinite_dataloader(
        train_data, config, Collate(config["device"])
    )
    val_loader = create_infinite_dataloader(val_data, config, Collate(config["device"]))

    return train_loader, val_loader
