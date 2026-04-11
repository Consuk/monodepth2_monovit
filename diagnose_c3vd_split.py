from __future__ import absolute_import, division, print_function

import argparse
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from datasets.c3vd_dataset import C3VDDataset
from utils import readlines, resolve_split_dir


def parse_line(line):
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid split line: {line}")
    folder = parts[0]
    frame_idx = int(parts[1])
    side = parts[2] if len(parts) > 2 else "l"
    return folder, frame_idx, side


def load_gray_small(path, size=(160, 128)):
    with Image.open(path) as im:
        im = im.convert("L").resize(size, Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def inspect_image_mode(path):
    with Image.open(path) as im:
        mode = im.mode
        size = im.size
    return mode, size


def main():
    parser = argparse.ArgumentParser("Diagnose C3VD split quality for Monodepth2 training")
    parser.add_argument("--data_path", type=str, required=True, help="C3VD dataset root")
    parser.add_argument("--split", type=str, default="c3vd", help="split folder name")
    parser.add_argument("--split_root", type=str, default=None, help="root containing split folders")
    parser.add_argument("--files_name", type=str, default="train_files.txt",
                        choices=["train_files.txt", "val_files.txt", "test_files.txt"])
    parser.add_argument("--sample_limit", type=int, default=-1, help="max samples to inspect (-1 = all)")
    parser.add_argument("--compute_motion", action="store_true",
                        help="compute quick grayscale abs-diff stats between t and neighbors")
    parser.add_argument("--inspect_modes", action="store_true",
                        help="inspect selected center image file modes to detect non-RGB mapping issues")
    parser.add_argument("--max_mode_checks", type=int, default=300,
                        help="max center images to inspect when --inspect_modes is set")
    opt = parser.parse_args()

    split_root = opt.split_root or os.path.join(os.path.dirname(__file__), "splits")
    split_dir = resolve_split_dir(opt.split, split_root)
    split_file = os.path.join(split_dir, opt.files_name)

    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"Missing split file: {split_file}")

    lines = readlines(split_file)
    if opt.sample_limit > 0:
        lines = lines[:opt.sample_limit]

    ds = C3VDDataset(
        data_path=opt.data_path,
        filenames=lines,
        height=256,
        width=320,
        frame_idxs=[0, -1, 1],
        num_scales=4,
        is_train=False,
        img_ext=".png",
    )

    total = len(lines)
    ok_center = 0
    ok_prev = 0
    ok_next = 0
    ok_triplet = 0
    bad_any = 0

    per_folder = defaultdict(lambda: {"n": 0, "miss_prev": 0, "miss_next": 0, "miss_center": 0})

    motion_prev = []
    motion_next = []

    suspicious_name = 0
    non_rgb_mode = 0
    checked_modes = 0
    suspicious_examples = []
    non_rgb_examples = []

    for i, line in enumerate(lines):
        folder, idx, side = parse_line(line)
        per_folder[folder]["n"] += 1

        center_path = None
        prev_path = None
        next_path = None

        try:
            center_path = ds.get_image_path(folder, idx, side)
            ok_center += 1
            lower = center_path.lower()
            if "depth" in lower or "normal" in lower or "flow" in lower or "occ" in lower:
                suspicious_name += 1
                if len(suspicious_examples) < 8:
                    suspicious_examples.append((i, center_path))

            if opt.inspect_modes and checked_modes < opt.max_mode_checks:
                mode, size = inspect_image_mode(center_path)
                checked_modes += 1
                if mode not in ("RGB", "RGBA"):
                    non_rgb_mode += 1
                    if len(non_rgb_examples) < 8:
                        non_rgb_examples.append((i, center_path, mode, size))
        except FileNotFoundError:
            per_folder[folder]["miss_center"] += 1

        try:
            prev_path = ds.get_image_path(folder, idx - 1, side)
            ok_prev += 1
        except FileNotFoundError:
            per_folder[folder]["miss_prev"] += 1

        try:
            next_path = ds.get_image_path(folder, idx + 1, side)
            ok_next += 1
        except FileNotFoundError:
            per_folder[folder]["miss_next"] += 1

        if center_path and prev_path and next_path:
            ok_triplet += 1
            if opt.compute_motion:
                c = load_gray_small(center_path)
                p = load_gray_small(prev_path)
                n = load_gray_small(next_path)
                motion_prev.append(float(np.mean(np.abs(c - p))))
                motion_next.append(float(np.mean(np.abs(c - n))))
        else:
            bad_any += 1

    print(f"[INFO] split file: {split_file}")
    print(f"[INFO] samples   : {total}")
    print("")
    print("=== Neighbor availability ===")
    print(f"center exists         : {ok_center}/{total} ({100.0 * ok_center / max(1, total):.2f}%)")
    print(f"prev (-1) exists      : {ok_prev}/{total} ({100.0 * ok_prev / max(1, total):.2f}%)")
    print(f"next (+1) exists      : {ok_next}/{total} ({100.0 * ok_next / max(1, total):.2f}%)")
    print(f"full (0,-1,+1) exists : {ok_triplet}/{total} ({100.0 * ok_triplet / max(1, total):.2f}%)")
    print(f"any missing neighbor  : {bad_any}/{total} ({100.0 * bad_any / max(1, total):.2f}%)")
    print(f"suspicious center path: {suspicious_name}/{total} ({100.0 * suspicious_name / max(1, total):.2f}%)")
    if opt.inspect_modes:
        print(f"non-RGB center modes  : {non_rgb_mode}/{checked_modes} ({100.0 * non_rgb_mode / max(1, checked_modes):.2f}%)")

    ranked = sorted(
        per_folder.items(),
        key=lambda kv: (kv[1]["miss_center"] + kv[1]["miss_prev"] + kv[1]["miss_next"]) / max(1, kv[1]["n"]),
        reverse=True,
    )
    print("")
    print("=== Worst folders (missing ratio) ===")
    for folder, st in ranked[:10]:
        misses = st["miss_center"] + st["miss_prev"] + st["miss_next"]
        ratio = 100.0 * misses / max(1, 3 * st["n"])
        print(
            f"{folder}: miss={misses}/{3 * st['n']} ({ratio:.2f}%), "
            f"n={st['n']}, miss_center={st['miss_center']}, miss_prev={st['miss_prev']}, miss_next={st['miss_next']}"
        )

    if suspicious_examples:
        print("")
        print("=== Suspicious center path examples ===")
        for i, p in suspicious_examples:
            print(f"idx={i}: {p}")

    if non_rgb_examples:
        print("")
        print("=== Non-RGB center mode examples ===")
        for i, p, mode, size in non_rgb_examples:
            print(f"idx={i}: mode={mode}, size={size}, path={p}")

    if opt.compute_motion and motion_prev and motion_next:
        mp = np.asarray(motion_prev, dtype=np.float32)
        mn = np.asarray(motion_next, dtype=np.float32)
        print("")
        print("=== Motion proxy (mean abs gray diff) ===")
        print(f"prev mean/median: {mp.mean():.5f} / {np.median(mp):.5f}")
        print(f"next mean/median: {mn.mean():.5f} / {np.median(mn):.5f}")
        print("Note: very tiny values can indicate low-motion pairs and weak self-supervision signal.")


if __name__ == "__main__":
    main()
