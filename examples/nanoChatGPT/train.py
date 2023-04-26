"""
Train the transformer model. Configurable via config/train.yaml, but any argument can
also be overridden at the command line.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False
"""

import os
import time
from pathlib import Path

import torch
from models.transformer import init_optimizer, init_transformer
from shared import create_lr_scheduler, setup
from tensordict.nn import TensorDictModule
from utils import load_and_update_config

from data.shakespeare import get_dataloaders

HERE = Path(__file__).parent


def init_scaler(config):
    # initialize a GradScaler. If enabled=False scaler is a no-op
    return torch.cuda.amp.GradScaler(enabled=(config["dtype"] == "float16"))


def create_loss_estimator(config):
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(model, dataloader):
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            batch = next(dataloader)
            with ctx:
                model(batch)
            losses[k] = batch.loss.item()
        return losses.mean()

    return estimate_loss


def train(config):
    # TODO: clean up...train should do just the training.
    # model creation, data loading etc. should be performed outside
    # plus align all script to have same structure and order of calls
    model, model_kwargs = init_transformer(config)
    model.to(config["device"])
    scaler = init_scaler(config)
    optimizer = init_optimizer(model, config)

    # compile the model
    if config["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    model = TensorDictModule(
        model, in_keys=["prompt", "target"], out_keys=["logits", "loss"]
    )

    # these will already have been set if resuming from previous checkpoint
    iter_num = config.setdefault("iter_num", 0)
    best_val_loss = config.setdefault("best_val_loss", 1e9)

    if config["decay_lr"]:
        lr_scheduler = create_lr_scheduler(config)
    else:

        def lr_scheduler(_):
            return config["learning_rate"]

    estimate_loss = create_loss_estimator(config)

    train_loader, val_loader = get_dataloaders(config)

    # training loop
    next_batch = next(train_loader)  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module
    running_mfu = -1.0

    while True:
        # determine and set the learning rate for this iteration
        lr = lr_scheduler(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
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
                    print(f"saving checkpoint to {config['out_dir']}")
                    torch.save(checkpoint, os.path.join(config["out_dir"], "ckpt.pt"))
        if iter_num == 0 and config["eval_only"]:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for _ in range(config["gradient_accumulation_steps"]):
            batch = next_batch
            with ctx:
                model(batch)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            next_batch = next(train_loader)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(batch.loss).backward()
        # clip the gradient
        if config["grad_clip"] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config["log_interval"] == 0:
            # loss as float. note: this is a CPU-GPU sync point
            lossf = batch.loss.item()
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    config["batch_size"] * config["gradient_accumulation_steps"], dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
                f"mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > config["max_iters"]:
            break


if __name__ == "__main__":
    config = load_and_update_config("config/train.yaml")

    ctx = setup(config)
    train(config)
