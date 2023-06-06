from pathlib import Path

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from transformers import GPT2LMHeadModel

HERE = Path(__file__).parent
DEFAULT_VOCAB_SIZE = 50_304


def init_transformer(
    config, as_tensordictmodule=True, skip_compilation=False, inference=False
):
    model_kwargs = {
        "resid_pdrop": config["dropout"],
        "embd_pdrop": config["dropout"],
        "attn_pdrop": config["dropout"],
        "summary_first_dropout": config["dropout"],
        # "n_positions": 1024,
    }

    # TODO: do we need to support "scratch"
    # TODO: init_base_from redundant? replace with transformer_path which can either
    # be "gpt2" or a path to a checkpoint
    if config["init_base_from"] in ["scratch", "pretrained"]:
        model = GPT2LMHeadModel.from_pretrained(
            config["base_model"], return_dict=False, **model_kwargs
        )
        if config["init_base_from"] == "scratch":
            model.post_init()
    elif config["init_base_from"] == "resume":
        model = GPT2LMHeadModel.from_pretrained(config["out_dir"], return_dict=False)
    else:
        raise ValueError(f"option {config['init_base_from']=} not recognised")

    # crop down the model block size if desired, using model surgery
    # if config["block_size"] < model.config.n_positions:
    #     print(
    #         f"cropping model from block_size {model.config.n_positions} to {config['block_size']}"
    #     )
    #     crop_block_size(model, config["block_size"])
    # print_trainable_parameters(model)

    model.to(config["device"])
    # compile the model
    if not skip_compilation and config["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    if as_tensordictmodule:
        if inference:
            out_keys = [("transformer_data", "logits")]
        else:
            out_keys = ["loss", ("transformer_data", "logits")]

        model = TensorDictModule(
            model,
            in_keys={
                ("transformer_data", "input_ids"): "input_ids",
                ("transformer_data", "attention_mask"): "attention_mask",
                ("transformer_data", "labels"): "labels",
            },
            out_keys=out_keys,
        )
    return model
