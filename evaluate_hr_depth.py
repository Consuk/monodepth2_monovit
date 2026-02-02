from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np

import torch
from torch.utils.data import DataLoader

import datasets
import networks
from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions


def infer_num_ch_enc(encoder, in_chans, height, width, device):
    encoder.eval()
    with torch.no_grad():
        x = torch.zeros(1, in_chans, height, width, device=device)
        feats = encoder(x)
        if not isinstance(feats, (list, tuple)):
            raise RuntimeError("Encoder forward must return a list/tuple of feature maps.")
        return [int(f.shape[1]) for f in feats]


def evaluate(opt):
    assert opt.load_weights_folder is not None, "Please specify --load_weights_folder"

    device = torch.device("cpu" if opt.no_cuda else "cuda")

    # Load encoder weights
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    height = int(encoder_dict.get("height", opt.height))
    width = int(encoder_dict.get("width", opt.width))

    encoder = networks.mpvit_tiny(in_chans=3).to(device)
    # remove metadata keys
    encoder_state = {k: v for k, v in encoder_dict.items() if k not in ["height", "width"]}
    encoder.load_state_dict({k: v for k, v in encoder_state.items() if k in encoder.state_dict()}, strict=False)

    encoder.num_ch_enc = infer_num_ch_enc(encoder, 3, height, width, device)

    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, opt.scales).to(device)
    depth_decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    depth_decoder.load_state_dict(torch.load(depth_decoder_path, map_location=device))

    encoder.eval()
    depth_decoder.eval()

    # Dataset
    splits_dir = os.path.join(os.path.dirname(__file__), "splits", opt.eval_split)
    filenames = readlines(os.path.join(splits_dir, "test_files.txt"))

    dataset_dict = {
        "endovis": datasets.EndovisDataset,
        "hamlyn": datasets.HamlynDataset,
        "kitti": datasets.KITTIRAWDataset,
        "kitti_depth": datasets.KITTIDepthDataset,
        "kitti_test": datasets.KITTITestDataset,
        "kitti_odom": datasets.KITTIOdomDataset,
    }
    dataset_cls = dataset_dict.get(opt.dataset, datasets.HamlynDataset)

    dataset = dataset_cls(
        opt.data_path, filenames, height, width, [0], 4,
        is_train=False,
        img_ext=".png" if opt.png else ".jpg",
        strict_neighbors=getattr(opt, "hamlyn_strict_neighbors", False),
        neighbor_search_max=getattr(opt, "neighbor_search_max", 10)
    )

    loader = DataLoader(dataset, opt.eval_batch_size, False, num_workers=opt.num_workers,
                        pin_memory=True, drop_last=False)

    pred_disps = []
    with torch.no_grad():
        for inputs in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}

            feats = encoder(inputs[("color", 0, 0)])
            outputs = depth_decoder(feats)
            disp = outputs[("disp", 0)]
            disp = torch.nn.functional.interpolate(disp, [height, width], mode="bilinear", align_corners=False)
            pred_disps.append(disp.cpu().numpy())

    pred_disps = np.concatenate(pred_disps, axis=0)
    out_path = os.path.join(opt.load_weights_folder, "pred_disps.npy")
    np.save(out_path, pred_disps)
    print("Saved predicted disparities to:", out_path)


if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()
    evaluate(opt)
