# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy

from data import get_prompt_dataloader
from env import rollout
from models.actor_critic import init_actor_critic
from models.reward import init_reward_model

from transformers import GPT2Tokenizer
from utils import get_file_logger, setup, resolve_name_or_path
from omegaconf import OmegaConf




def main():
    cfg = OmegaConf.load('config/train_rlhf.yaml')
    data_cfg = cfg.data
    data_cfg.batch_size = 16
    model_cfg = cfg.model
    reward_model_cfg = cfg.reward_model
    dropout = model_cfg.dropout


    device = cfg.sys.device
    compile_ = cfg.sys.compile

    val_loader = get_prompt_dataloader(data_cfg, device=device, split="valid")

    _, _, _, model = init_actor_critic(
        resolve_name_or_path("./out_rlhf"), dropout, device, compile_
    )
    ref_model = deepcopy(model).to("cuda:1")
    ref_model.eval()
    ref_model.requires_grad_(False)
    model.eval()
    model.requires_grad_(False)

    reward_model = init_reward_model(
        reward_model_path=resolve_name_or_path(reward_model_cfg.name_or_path),
        device=device,
        compile_=compile_,
    )
    reward_model.eval()
    reward_model.requires_grad_(False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    test_batch = next(val_loader)
    td = rollout(
        test_batch, model, ref_model, reward_model, max_new_tokens=50, kl_coef=0
    )
    rewards = td.get(("next", "reward")).sum(dim=1).mean().item()
    responses = td.get(("next", "input_ids"))

    for idx in range(len(responses)):
        test_reward = td.get(("next", "reward"))[idx:idx+1].sum(dim=1).mean().item()
        test_rindex = test_batch.prompt_rindex[idx]
        test_prompt_ids = test_batch.input_ids[idx:idx+1, :test_rindex]
        test_label_ids = test_batch.input_ids[idx:idx+1, test_rindex:]
        response_ids = responses[idx, -1, test_rindex:]    
        response = tokenizer.decode(
            response_ids[response_ids != tokenizer.eos_token_id].tolist()
        )

        test_prompt = tokenizer.decode(test_prompt_ids[0, :test_rindex].tolist())
        test_label = tokenizer.decode(
            test_label_ids[0, test_label_ids[0] != tokenizer.pad_token_id].tolist()
        )

        string_to_write = (
            f"Query:\n{test_prompt}\n"
            f"Response:\n{response}\n"
            f"Actual response:\n{test_label}\n"
            f"{test_reward=:4.4f}, "
            f"{rewards=:4.4f}, "
            f"====================================================\n"
        )
        print(string_to_write)


if __name__ == "__main__":
    main()
