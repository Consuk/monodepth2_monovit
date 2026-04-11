# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()

import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85, 0)  # Use only 85% of GPU memory


def init_wandb(opt):
    try:
        import wandb
    except Exception:
        return None

    kwargs = {
        "project": opt.wandb_project,
        "mode": opt.wandb_mode,
        "config": vars(opt),
    }
    if opt.wandb_entity:
        kwargs["entity"] = opt.wandb_entity
    if opt.wandb_name:
        kwargs["name"] = opt.wandb_name
    if opt.wandb_tags:
        kwargs["tags"] = opt.wandb_tags

    return wandb.init(**kwargs)


if __name__ == "__main__":
    run = init_wandb(opts)
    try:
        trainer = Trainer(opts)
        trainer.train()
    finally:
        if run is not None:
            run.finish()

