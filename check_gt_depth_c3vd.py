from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import numpy as np

from export_gt_depth import parse_split_line, find_depth_file, load_depth
from utils import readlines, resolve_split_dir


def main():
    parser = argparse.ArgumentParser("Check exported C3VD gt_depths.npz correctness")
    parser.add_argument("--data_path", type=str, required=True, help="C3VD dataset root")
    parser.add_argument("--split", type=str, default="c3vd", help="split name")
    parser.add_argument(
        "--split_root",
        type=str,
        default=None,
        help="root containing split folders (default: <repo>/splits)",
    )
    parser.add_argument(
        "--files_name",
        type=str,
        default="test_files.txt",
        choices=["train_files.txt", "val_files.txt", "test_files.txt"],
        help="which split file to check against",
    )
    parser.add_argument(
        "--npz_name",
        type=str,
        default="gt_depths.npz",
        help="name of npz file under split directory",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="max number of samples to check (-1 = all)",
    )
    parser.add_argument(
        "--tol_abs",
        type=float,
        default=1e-4,
        help="absolute error tolerance against decoded source depth",
    )
    opt = parser.parse_args()

    default_split_root = os.path.join(os.path.dirname(__file__), "splits")
    split_root = opt.split_root or default_split_root
    split_dir = resolve_split_dir(opt.split, split_root)

    file_list_path = os.path.join(split_dir, opt.files_name)
    npz_path = os.path.join(split_dir, opt.npz_name)

    if not os.path.isfile(file_list_path):
        print(f"[ERROR] split file not found: {file_list_path}")
        return 2
    if not os.path.isfile(npz_path):
        print(f"[ERROR] npz not found: {npz_path}")
        return 2

    lines = readlines(file_list_path)
    data_npz = np.load(npz_path, allow_pickle=True)
    if "data" in data_npz:
        gt_depths = data_npz["data"]
    else:
        first_key = list(data_npz.keys())[0]
        gt_depths = data_npz[first_key]

    if isinstance(gt_depths, np.ndarray) and gt_depths.dtype == object:
        gt_depths = list(gt_depths)

    n_lines = len(lines)
    n_gt = len(gt_depths)
    print(f"[INFO] split dir : {split_dir}")
    print(f"[INFO] split file: {file_list_path} ({n_lines} lines)")
    print(f"[INFO] gt npz    : {npz_path} ({n_gt} maps)")

    if n_lines != n_gt:
        print(f"[FAIL] Count mismatch: lines={n_lines}, gt_maps={n_gt}")
        return 1

    n_check = n_lines if opt.max_samples < 0 else min(n_lines, opt.max_samples)
    print(f"[INFO] checking  : {n_check} samples")

    bad_shape = 0
    bad_nonfinite = 0
    bad_range = 0
    missing_source = 0
    decode_mismatch = 0

    global_min = float("inf")
    global_max = float("-inf")
    global_mae = []
    global_max_abs = []

    for i in range(n_check):
        gt = np.asarray(gt_depths[i], dtype=np.float32)
        if gt.ndim != 2:
            bad_shape += 1
            print(f"[FAIL] idx={i}: gt depth is not 2D, shape={gt.shape}")
            continue

        if not np.isfinite(gt).all():
            bad_nonfinite += 1
            print(f"[FAIL] idx={i}: non-finite values detected")
            continue

        mn = float(gt.min())
        mx = float(gt.max())
        global_min = min(global_min, mn)
        global_max = max(global_max, mx)

        if mn < -1e-6 or mx > 100.0 + 1e-3:
            bad_range += 1
            print(f"[FAIL] idx={i}: value range out of expected [0,100] mm, min={mn:.6f}, max={mx:.6f}")

        folder, frame_token = parse_split_line(lines[i])
        src_path = find_depth_file(opt.data_path, opt.split, folder, frame_token)
        if src_path is None:
            missing_source += 1
            print(f"[WARN] idx={i}: source depth file not found for '{lines[i]}'")
            continue

        src_depth = load_depth(opt.split, src_path).astype(np.float32)
        if src_depth.shape != gt.shape:
            decode_mismatch += 1
            print(
                f"[FAIL] idx={i}: shape mismatch vs source decode, "
                f"gt={gt.shape}, src={src_depth.shape}, file={src_path}"
            )
            continue

        diff = np.abs(gt - src_depth)
        mae = float(diff.mean())
        max_abs = float(diff.max())
        global_mae.append(mae)
        global_max_abs.append(max_abs)

        if max_abs > opt.tol_abs:
            decode_mismatch += 1
            print(
                f"[FAIL] idx={i}: decoded mismatch, mae={mae:.8f}, "
                f"max_abs={max_abs:.8f}, tol={opt.tol_abs:.8f}, file={src_path}"
            )

    print("\n=== Summary ===")
    print(f"checked samples         : {n_check}")
    print(f"bad shape               : {bad_shape}")
    print(f"non-finite maps         : {bad_nonfinite}")
    print(f"out-of-range maps       : {bad_range}")
    print(f"missing source files    : {missing_source}")
    print(f"decode mismatches       : {decode_mismatch}")

    if global_min == float("inf"):
        print("global min / max        : n/a")
    else:
        print(f"global min / max (mm)   : {global_min:.6f} / {global_max:.6f}")

    if global_mae:
        print(f"mean MAE (mm)           : {np.mean(global_mae):.10f}")
        print(f"max abs error (mm)      : {np.max(global_max_abs):.10f}")
    else:
        print("mean MAE / max abs      : n/a")

    failed = (bad_shape + bad_nonfinite + bad_range + decode_mismatch) > 0
    if failed:
        print("[RESULT] FAIL")
        return 1

    if missing_source > 0:
        print("[RESULT] PASS with warnings (some source files not found)")
        return 0

    print("[RESULT] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
