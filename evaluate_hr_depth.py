from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets
import networks
from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions


def evaluate(opt):
    assert opt.load_weights_folder is not None, "Please specify --load_weights_folder"

    device = torch.device("cpu" if opt.no_cuda else "cuda")

    # ------------------------------------------------------------------
    # Load encoder weights and recover training resolution
    # ------------------------------------------------------------------
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    if not os.path.isfile(encoder_path):
        raise FileNotFoundError(f"Cannot find encoder weights: {encoder_path}")
    if not os.path.isfile(decoder_path):
        raise FileNotFoundError(f"Cannot find depth decoder weights: {decoder_path}")

    encoder_dict = torch.load(encoder_path, map_location=device)
    height = int(encoder_dict.get("height", opt.height))
    width = int(encoder_dict.get("width", opt.width))

    # ------------------------------------------------------------------
    # Rebuild the SAME architecture used during training
    # trainer.py uses mpvit_small + standard DepthDecoder
    # ------------------------------------------------------------------
    encoder = networks.mpvit_small().to(device)
    encoder.num_ch_enc = [64, 128, 216, 288, 288]

    encoder_state = {k: v for k, v in encoder_dict.items() if k not in ["height", "width"]}
    model_dict = encoder.state_dict()
    filtered_encoder_state = {k: v for k, v in encoder_state.items() if k in model_dict}
    model_dict.update(filtered_encoder_state)
    encoder.load_state_dict(model_dict)

    depth_decoder = networks.DepthDecoder(
        encoder.num_ch_enc,
        scales=opt.scales
    ).to(device)
    depth_decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encoder.eval()
    depth_decoder.eval()

    # ------------------------------------------------------------------
    # Dataset / split
    # ------------------------------------------------------------------
    splits_dir = os.path.join(os.path.dirname(__file__), "splits", opt.eval_split)
    test_file = os.path.join(splits_dir, "test_files.txt")
    if not os.path.isfile(test_file):
        raise FileNotFoundError(f"Cannot find split file: {test_file}")

    filenames = readlines(test_file)

    dataset_dict = {
        "endovis": getattr(datasets, "EndovisDataset", None),
        "hamlyn": getattr(datasets, "HamlynDataset", None),
        "kitti": getattr(datasets, "KITTIRAWDataset", None),
        "kitti_depth": getattr(datasets, "KITTIDepthDataset", None),
        "kitti_test": getattr(datasets, "KITTITestDataset", None),
        "kitti_odom": getattr(datasets, "KITTIOdomDataset", None),
    }

    if opt.dataset not in dataset_dict or dataset_dict[opt.dataset] is None:
        raise KeyError(f"Dataset '{opt.dataset}' is not available in datasets package")

    dataset_cls = dataset_dict[opt.dataset]

    dataset = dataset_cls(
        opt.data_path,
        filenames,
        height,
        width,
        [0],
        4,
        is_train=False,
        img_ext=".png" if opt.png else ".jpg",
        strict_neighbors=getattr(opt, "hamlyn_strict_neighbors", False),
        neighbor_search_max=getattr(opt, "neighbor_search_max", 10)
    )

    loader = DataLoader(
        dataset,
        batch_size=opt.eval_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=not opt.no_cuda,
        drop_last=False
    )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    pred_disps = []
    pred_depths = []

    print("Evaluating with:")
    print(f"  dataset     : {opt.dataset}")
    print(f"  eval_split  : {opt.eval_split}")
    print(f"  min_depth   : {opt.min_depth}")
    print(f"  max_depth   : {opt.max_depth}")
    print(f"  input size  : {height}x{width}")
    print(f"  num samples : {len(filenames)}")

    with torch.no_grad():
        for inputs in loader:
            inputs = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }

            features = encoder(inputs[("color", 0, 0)])
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp = F.interpolate(
                disp,
                [height, width],
                mode="bilinear",
                align_corners=False
            )

            _, depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)

            pred_disps.append(disp.cpu().numpy())
            pred_depths.append(depth.cpu().numpy())

    pred_disps = np.concatenate(pred_disps, axis=0)
    pred_depths = np.concatenate(pred_depths, axis=0)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    disp_out_path = os.path.join(opt.load_weights_folder, "pred_disps.npy")
    depth_out_path = os.path.join(
        opt.load_weights_folder,
        f"pred_depths_min{opt.min_depth:g}_max{opt.max_depth:g}.npy"
    )

    np.save(disp_out_path, pred_disps)
    np.save(depth_out_path, pred_depths)

    print("Saved predicted disparities to:", disp_out_path)
    print("Saved predicted depths to     :", depth_out_path)
    print("Done.")


if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()
    evaluate(opt)