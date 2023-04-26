from pathlib import Path

from .transformer import init_model
from .rlhf import RLHF
from .utils import _remove_state_dict_prefixes, load_checkpoint


def init_reward_model(config):
    # FIXME: Don't like this. include it into model
    model, model_kwargs = init_model(config)
    model = RLHF(model, mode="reward", discrete_reward=False)

    print("Config of model: ", model.config)
    out_dir = Path(config["out_dir_reward"])

    if not out_dir.exists(config["out_dir_reward"]):
        print(f"Create {config['out_dir_reward']}")
        out_dir.mkdir()

    if config["init_multihead_from"] == "scratch":
        print("initializing multihead from scratch")
    elif config["init_multihead_from"] == "resume":
        print(f"Resuming training from {config['out_dir_reward']}")
        checkpoint = load_checkpoint(out_dir, device=config["device"])
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        _remove_state_dict_prefixes(state_dict)
        model.load_state_dict(state_dict)

    return model, model_kwargs
