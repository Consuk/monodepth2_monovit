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
torch.cuda.set_per_process_memory_fraction(0.85, 0)  # Use only 85% of GPU memory


if __name__ == "__main__":
    import wandb
    wandb.init(project="Monodepth2_monovit")
    trainer = Trainer(opts)
    trainer.train()

