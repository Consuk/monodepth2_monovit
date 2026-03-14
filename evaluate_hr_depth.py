from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets
import networks
from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions

cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    eps = 1e-6
    gt = np.clip(gt, eps, None)
    pred = np.clip(pred, eps, None)

    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def build_eval_dataset(opt, filenames, height, width):
    img_ext = ".png" if bool(getattr(opt, "png", False)) else ".jpg"

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

    DatasetClass = dataset_dict[opt.dataset]

    dataset = DatasetClass(
        opt.data_path,
        filenames,
        height,
        width,
        [0],
        4,
        is_train=False,
        img_ext=img_ext,
        strict_neighbors=getattr(opt, "hamlyn_strict_neighbors", False),
        neighbor_search_max=getattr(opt, "neighbor_search_max", 10)
    )
    return dataset


def evaluate(opt):
    assert opt.load_weights_folder is not None, "Please specify --load_weights_folder"
    assert sum((bool(getattr(opt, "eval_mono", False)),
                bool(getattr(opt, "eval_stereo", False)))) == 1, \
        "Choose mono or stereo evaluation: set exactly one of eval_mono/eval_stereo"

    device = torch.device("cpu" if opt.no_cuda else "cuda")
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    if not os.path.isdir(opt.load_weights_folder):
        raise FileNotFoundError(f"Cannot find folder: {opt.load_weights_folder}")

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    if not os.path.isfile(encoder_path):
        raise FileNotFoundError(f"Cannot find encoder weights: {encoder_path}")
    if not os.path.isfile(decoder_path):
        raise FileNotFoundError(f"Cannot find depth decoder weights: {decoder_path}")

    # ------------------------------------------------------------------
    # Rebuild SAME architecture used in trainer.py:
    # mpvit_small + standard DepthDecoder
    # ------------------------------------------------------------------
    encoder_dict = torch.load(encoder_path, map_location=device)
    height = int(encoder_dict.get("height", opt.height))
    width = int(encoder_dict.get("width", opt.width))

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
    # Filenames / split
    # ------------------------------------------------------------------
    custom_list = getattr(opt, "eval_filelist", None)
    if custom_list is not None:
        custom_list = os.path.expanduser(custom_list)
        print(f"-> Using custom eval file list: {custom_list}")
        filenames = readlines(custom_list)
    else:
        split_file = os.path.join(splits_dir, opt.eval_split, "test_files.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Cannot find split file: {split_file}")
        filenames = readlines(split_file)

    dataset = build_eval_dataset(opt, filenames, height, width)

    dataloader = DataLoader(
        dataset,
        int(getattr(opt, "eval_batch_size", 16)),
        shuffle=False,
        num_workers=int(getattr(opt, "num_workers", 4)),
        pin_memory=not opt.no_cuda,
        drop_last=False
    )

    print("Evaluating with:")
    print(f"  dataset     : {opt.dataset}")
    print(f"  eval_split  : {opt.eval_split}")
    print(f"  min_depth   : {opt.min_depth}")
    print(f"  max_depth   : {opt.max_depth}")
    print(f"  input size  : {height}x{width}")
    print(f"  num samples : {len(filenames)}")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    pred_disps = []
    pred_depths = []

    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].to(device, non_blocking=True)

            if bool(getattr(opt, "post_process", False)):
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            features = encoder(input_color)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp = F.interpolate(
                disp,
                [height, width],
                mode="bilinear",
                align_corners=False
            )

            scaled_disp, depth = disp_to_depth(
                disp,
                float(getattr(opt, "min_depth", 1e-3)),
                float(getattr(opt, "max_depth", 150.0))
            )

            pred_disps.append(scaled_disp.cpu()[:, 0].numpy())
            pred_depths.append(depth.cpu()[:, 0].numpy())

    pred_disps = np.concatenate(pred_disps, axis=0)
    pred_depths = np.concatenate(pred_depths, axis=0)

    disp_out_path = os.path.join(opt.load_weights_folder, "pred_disps.npy")
    depth_out_path = os.path.join(
        opt.load_weights_folder,
        f"pred_depths_min{opt.min_depth:g}_max{opt.max_depth:g}.npy"
    )

    np.save(disp_out_path, pred_disps)
    np.save(depth_out_path, pred_depths)

    print("Saved predicted disparities to:", disp_out_path)
    print("Saved predicted depths to     :", depth_out_path)

    if bool(getattr(opt, "no_eval", False)):
        print("-> Evaluation disabled. Done.")
        return {}

    # ------------------------------------------------------------------
    # Load GT depths
    # ------------------------------------------------------------------
    custom_gt = getattr(opt, "gt_depths_path", None)
    if custom_gt is not None:
        gt_path = os.path.expanduser(custom_gt)
        print(f"-> Using custom gt_depths.npz: {gt_path}")
    else:
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"gt_depths.npz not found: {gt_path}")

    data_npz = np.load(gt_path, fix_imports=True, encoding="latin1", allow_pickle=True)
    gt_depths = data_npz["data"]

    if isinstance(gt_depths, np.ndarray) and gt_depths.dtype == object:
        gt_depths = list(gt_depths)

    num_pred = pred_disps.shape[0]
    num_gt = len(gt_depths)
    assert num_pred == num_gt, f"Mismatch: {num_pred} predictions vs {num_gt} gt depth maps"

    MIN_DEPTH = float(getattr(opt, "min_depth", 1e-3))
    MAX_DEPTH = float(getattr(opt, "max_depth", 150.0))

    if bool(getattr(opt, "eval_stereo", False)):
        print(f"   Stereo evaluation - scaling by {STEREO_SCALE_FACTOR}")
        disable_median_scaling = True
        pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")
        disable_median_scaling = bool(getattr(opt, "disable_median_scaling", False))
        pred_depth_scale_factor = float(getattr(opt, "pred_depth_scale_factor", 1.0))

    errors = []
    ratios = []

    for i in range(num_pred):
        gt_depth = np.asarray(gt_depths[i])
        if gt_depth.dtype == object:
            gt_depth = gt_depth.astype(np.float32)
        else:
            gt_depth = gt_depth.astype(np.float32, copy=False)

        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp_resized = cv2.resize(
            pred_disp,
            (gt_width, gt_height),
            interpolation=cv2.INTER_LINEAR
        )

        # IMPORTANT:
        # pred_disps already stores scaled disparity from disp_to_depth(...)
        pred_depth = 1.0 / np.maximum(pred_disp_resized, 1e-6)

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        gt = gt_depth[mask]
        pred_depth = pred_depth[mask] * pred_depth_scale_factor

        gt = np.asarray(gt, dtype=np.float32)
        pred_depth = np.asarray(pred_depth, dtype=np.float32)

        if gt.size == 0:
            continue

        if not disable_median_scaling:
            ratio = np.median(gt) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth = np.clip(pred_depth, MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_errors(gt, pred_depth))

    if len(errors) == 0:
        raise RuntimeError("No valid samples found for evaluation after masking.")

    if (not disable_median_scaling) and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(f" Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")

    mean_errors = np.array(errors).mean(0)

    metrics = {
        "abs_rel": float(mean_errors[0]),
        "sq_rel": float(mean_errors[1]),
        "rmse": float(mean_errors[2]),
        "rmse_log": float(mean_errors[3]),
        "a1": float(mean_errors[4]),
        "a2": float(mean_errors[5]),
        "a3": float(mean_errors[6]),
    }

    print("\n  " + ("{:>8} | " * 7).format(
        "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"
    ))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("-> Done!")

    return metrics


if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()
    evaluate(opt)