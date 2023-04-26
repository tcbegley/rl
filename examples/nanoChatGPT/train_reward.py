import os
from pathlib import Path

import torch
from models.reward import init_reward_model
from shared import create_lr_scheduler, setup
from tensordict.nn import TensorDictModule
from utils import load_and_update_config

from data.openai_summarize_comparisons import get_dataloaders

HERE = Path(__file__).parent


# helps estimate an arbitrarily accurate loss over either split using many batches
def create_loss_estimator(config):
    @torch.no_grad()
    def estimate_loss(model, dataloader):
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            batch = next(dataloader)
            with ctx:
                reward_chosen = model(batch.chosen)
                reward_rejected = model(batch.rejected)
                loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
            losses[k] = loss.item()
        return losses.mean()

    return estimate_loss


def train_reward_model(config):
    # TODO: clean up...train should do just the training.
    # model creation, data loading etc. should be performed outside
    # plus align all script to have same structure and order of calls

    # GET DATA
    train_loader, val_loader = get_dataloaders(config)

    model, model_kwargs = init_reward_model(config)
    model.to(config["device"])

    # compile the model
    if config["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    model = TensorDictModule(model, in_keys=["input"], out_keys=["reward"])
    # FIXME: which one?
    # optimizer = torch.optim.AdamW(model.model.reward_head.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)

    # training loop
    local_iter_num = 0  # number of iterations in the lifetime of this process
    config["running_mfu"] = -1.0
    raw_model = model.module
    loss = None
    # these will already have been set if resuming from previous checkpoint
    iter_num = config.setdefault("iter_num", 0)
    best_val_loss = config.setdefault("best_val_loss", 1e9)

    estimate_loss = create_loss_estimator(config)

    if config["decay_lr"]:
        lr_scheduler = create_lr_scheduler(config)
    else:

        def lr_scheduler(_):
            return config["learning_rate"]

    while True:
        # determine and set the learning rate for this iteration
        lr = lr_scheduler(iter_num)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # # every once in a while evaluate the loss on train and val sets
        if iter_num % config["eval_interval"] == 0:
            model.eval()
            losses = {
                "train": estimate_loss(model, train_loader),
                "val": estimate_loss(model, val_loader),
            }
            model.train()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss or config["always_save_checkpoint"]:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_kwargs": model_kwargs,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {config['out_dir_reward']}")
                    torch.save(
                        checkpoint, os.path.join(config["out_dir_reward"], "ckpt.pt")
                    )
        if iter_num == 0 and config["eval_only"]:
            break

        batch = next(train_loader)

        # TODO: check why is different from std model (missing micro gradients)

        # TODO: combine evaluate_loss function with this. it's almost the same thing
        # evaluate the loss
        reward_chosen = model(batch.chosen)
        reward_rejected = model(batch.rejected)
        loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > config["max_iters"]:
            break


if __name__ == "__main__":
    config = load_and_update_config("config/train_reward.yaml")

    ctx = setup(config)
    train_reward_model(config)
