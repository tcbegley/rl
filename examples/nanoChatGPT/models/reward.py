from pathlib import Path

from .transformer import init_transformer
from .rlhf import RLHF
from .utils import _remove_state_dict_prefixes, load_checkpoint


def init_reward_model(config):
    # FIXME: Don't like this. include it into model
    model, model_kwargs = init_transformer(config)
    model = RLHF(model, mode="reward", discrete_reward=False)

    print("Config of model: ", model.config)
    out_dir = Path(config["out_dir_reward"])

    if not out_dir.exists():
        print(f"Create {config['out_dir_reward']}")
        out_dir.mkdir()

    if config["init_multihead_from"] == "scratch":
        print("initializing multihead from scratch")
    elif config["init_multihead_from"] == "resume":
        print(f"Resuming training from {config['out_dir_reward']}")
        checkpoint = load_checkpoint(out_dir, device=config["device"])
        state_dict = checkpoint["model"]
        _remove_state_dict_prefixes(state_dict)
        model.load_state_dict(state_dict)

    return model, model_kwargs


if __name__ == "__main__":
    import tiktoken
    import torch

    # FIXME: this relative import breaks when running this file
    # below code gives an example of usage but is not easily runnable
    from .utils import load_and_update_config

    enc = tiktoken.get_encoding("gpt2")

    HERE = Path(__file__).parent
    config = load_and_update_config(HERE.parent / "config" / "train_reward.yaml")
    reward_model = init_reward_model(config)

    prompt = enc.encode("this is a hard-coded prompt!")
    # add singleton leading dimension to simulate batch dimension
    prompt = torch.tensor(prompt)[None, :]

    reward = reward_model.forward_reward(prompt)
    print(reward)
