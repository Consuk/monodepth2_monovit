from __future__ import absolute_import, division, print_function

import argparse
import glob
import os

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from PIL import Image

from utils import readlines, resolve_split_dir


def parse_split_line(line):
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(
            "Invalid split line. Expected '<folder> <frame_idx> l', got: "
            f"{line}"
        )
    folder = parts[0]
    frame_token = parts[1]
    return folder, frame_token


def frame_tokens(frame_token):
    tokens = [str(frame_token)]
    try:
        n = int(frame_token)
        for w in (3, 4, 5, 6, 7, 8, 9, 10):
            tokens.append(f"{n:0{w}d}")
    except (TypeError, ValueError):
        pass

    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def candidate_depth_dirs(folder):
    return [
        folder,
        os.path.join(folder, "depth"),
        os.path.join(folder, "depths"),
        os.path.join(folder, "depth_map"),
        os.path.join(folder, "depth_maps"),
        os.path.join(folder, "gt_depth"),
        os.path.join(folder, "gt_depths"),
        os.path.join(folder, "Ground_truth_CT", "DepthL"),
    ]


def candidate_depth_names(frame_token):
    names = []
    suffixes = ["", "_depth", "-depth"]
    exts = [".png", ".tif", ".tiff", ".npy"]
    for token in frame_tokens(frame_token):
        for suffix in suffixes:
            for ext in exts:
                names.append(f"{token}{suffix}{ext}")

    seen = set()
    unique = []
    for n in names:
        if n not in seen:
            unique.append(n)
            seen.add(n)
    return unique


def find_depth_file(data_path, split_name, folder, frame_token):
    split_name = split_name.lower()
    roots = [data_path]
    if split_name in ("serv-ct", "endovis"):
        roots.append(os.path.join(data_path, "SERV-CT"))

    for root in roots:
        for rel_dir in candidate_depth_dirs(folder):
            abs_dir = os.path.join(root, rel_dir)
            for name in candidate_depth_names(frame_token):
                path = os.path.join(abs_dir, name)
                if os.path.isfile(path):
                    return path

    # Last-resort recursive search under likely roots.
    for root in roots:
        for token in frame_tokens(frame_token):
            for ext in ("png", "tif", "tiff", "npy"):
                hits = glob.glob(
                    os.path.join(root, "**", f"{token}*.{ext}"),
                    recursive=True,
                )
                if hits:
                    hits.sort()
                    return hits[0]

    return None


def decode_depth(split_name, raw):
    split_name = split_name.lower()
    arr = np.asarray(raw)

    if arr.ndim == 3:
        arr = arr[:, :, 0]

    if split_name == "c3vd":
        # C3VD convention: uint16 depth in [0, 65535] mapped to 0..100 mm.
        if arr.dtype == np.uint16:
            depth_mm = arr.astype(np.float32) * (100.0 / 65535.0)
        else:
            depth_mm = arr.astype(np.float32)
        depth_mm = np.clip(depth_mm, 0.0, 100.0)
        return depth_mm

    # Legacy fallback for other endoscopy-style depth PNGs.
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 256.0
    return arr.astype(np.float32)


def load_depth(split_name, path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        raw = np.load(path)
    else:
        if cv2 is not None:
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            raw = np.array(Image.open(path))
        if raw is None:
            raise RuntimeError(f"Could not read depth file: {path}")
    return decode_depth(split_name, raw)


def export_gt_depths():
    parser = argparse.ArgumentParser(description="Export gt_depths.npz for a split")
    parser.add_argument("--data_path", type=str, required=True, help="dataset root path")
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["eigen", "eigen_benchmark", "hamlyn", "SERV-CT", "endovis", "c3vd"],
        help="split name",
    )
    parser.add_argument(
        "--split_root",
        type=str,
        default=None,
        help="optional root directory that contains split folders (default: <repo>/splits)",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="skip missing depth files instead of failing",
    )
    opt = parser.parse_args()

    default_split_root = os.path.join(os.path.dirname(__file__), "splits")
    split_root = opt.split_root or default_split_root
    split_dir = resolve_split_dir(opt.split, split_root)

    test_files = os.path.join(split_dir, "test_files.txt")
    if not os.path.isfile(test_files):
        raise FileNotFoundError(f"Missing split file: {test_files}")

    lines = readlines(test_files)
    gt_depths = []
    missing = []

    print(f"Exporting GT depths for split '{opt.split}' from: {split_dir}")

    for idx, line in enumerate(lines):
        folder, frame_token = parse_split_line(line)
        depth_path = find_depth_file(opt.data_path, opt.split, folder, frame_token)

        if depth_path is None:
            missing.append((idx, line))
            if not opt.allow_missing:
                raise FileNotFoundError(
                    f"Depth file not found for line {idx}: '{line}' under data_path={opt.data_path}"
                )
            continue

        depth = load_depth(opt.split, depth_path)
        gt_depths.append(depth.astype(np.float32))

    if len(gt_depths) == 0:
        raise RuntimeError("No depth maps were exported.")

    output_path = os.path.join(split_dir, "gt_depths.npz")
    np.savez_compressed(output_path, data=np.array(gt_depths, dtype=object))

    print(f"Saved: {output_path}")
    print(f"Exported depth maps: {len(gt_depths)} / {len(lines)}")

    if missing:
        print(f"Missing entries: {len(missing)}")
        for i, line in missing[:10]:
            print(f"  - idx={i}: {line}")


if __name__ == "__main__":
    export_gt_depths()
