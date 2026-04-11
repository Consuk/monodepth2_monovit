from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import random
import re
from collections import defaultdict


KNOWN_IMAGE_DIRS = {
    "rgb",
    "images",
    "image",
    "color",
    "data",
    "left",
    "left_rectified",
}


def list_image_files(root, exts):
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                hits.append(os.path.join(dirpath, name))
    return hits


def parse_frame_index_from_name(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    stem_l = stem.lower()
    is_color_like = (
        stem_l.endswith("_color")
        or stem_l.endswith("-color")
        or stem.isdigit()
    )
    if not is_color_like:
        return None

    stem = stem.replace("_color", "").replace("-color", "")

    if stem.isdigit():
        return int(stem)

    groups = re.findall(r"\d+", stem)
    if not groups:
        return None
    return int(groups[-1])


def sequence_root_for_image(image_path, data_path):
    rel = os.path.relpath(image_path, data_path).replace("\\", "/")
    rel_dir = os.path.dirname(rel)
    if not rel_dir:
        return "."

    parts = rel_dir.split("/")
    if parts[-1].lower() in KNOWN_IMAGE_DIRS and len(parts) >= 2:
        return "/".join(parts[:-1])
    return rel_dir


def discover_sequences(data_path, exts):
    seq_to_frames = defaultdict(set)
    for image_path in list_image_files(data_path, exts):
        frame_idx = parse_frame_index_from_name(image_path)
        if frame_idx is None:
            continue
        seq_root = sequence_root_for_image(image_path, data_path)
        seq_to_frames[seq_root].add(frame_idx)
    return {k: sorted(v) for k, v in seq_to_frames.items()}


def discover_sequences_under_subdir(data_path, split_subdir, exts):
    base = os.path.join(data_path, split_subdir)
    if not os.path.isdir(base):
        return {}
    seq_to_frames = discover_sequences(base, exts)
    out = {}
    for seq_rel, frames in seq_to_frames.items():
        if seq_rel == ".":
            full_rel = split_subdir
        else:
            full_rel = f"{split_subdir}/{seq_rel}".replace("//", "/")
        out[full_rel] = frames
    return out


def discover_sequences_under_aliases(data_path, aliases, exts):
    for name in aliases:
        seq_map = discover_sequences_under_subdir(data_path, name, exts)
        if seq_map:
            return seq_map, name

    for name in aliases:
        base = os.path.join(data_path, name)
        if os.path.isdir(base):
            return discover_sequences_under_subdir(data_path, name, exts), name

    return {}, None


def build_lines_for_folders(seq_to_frames, folders, side):
    lines = []
    for folder in folders:
        frames = seq_to_frames.get(folder, [])
        for frame_idx in frames:
            lines.append(f"{folder} {frame_idx} {side}")
    return lines


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def can_write_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        return False

    probe = os.path.join(path, ".codex_write_probe")
    try:
        with open(probe, "w") as f:
            f.write("ok\n")
        os.remove(probe)
        return True
    except Exception:
        return False


def choose_output_dir(preferred):
    if can_write_directory(preferred):
        return preferred

    fallback = os.path.join("splits", "endovis", "c3vd")
    if can_write_directory(fallback):
        print(
            f"[warn] Cannot write to '{preferred}'. Falling back to '{fallback}'."
        )
        return fallback

    raise PermissionError(
        f"Cannot write split files to '{preferred}' or fallback '{fallback}'."
    )


def protocol_folder_layout(args, exts):
    train_map, train_name = discover_sequences_under_aliases(
        args.data_path, ["train", "training"], exts
    )
    val_map, val_name = discover_sequences_under_aliases(
        args.data_path, ["val", "validation"], exts
    )
    test_map, test_name = discover_sequences_under_aliases(
        args.data_path, ["test", "testing"], exts
    )

    print(
        "folder_layout roots -> "
        f"train: {train_name or '(missing)'}, "
        f"val: {val_name or '(missing)'}, "
        f"test: {test_name or '(missing)'}"
    )

    train_lines = build_lines_for_folders(train_map, sorted(train_map.keys()), args.side)
    val_lines = build_lines_for_folders(val_map, sorted(val_map.keys()), args.side)
    test_lines = build_lines_for_folders(test_map, sorted(test_map.keys()), args.side)
    return train_lines, val_lines, test_lines


def split_folders_disjoint(folders, train_ratio, val_ratio, seed):
    folders = list(folders)
    rng = random.Random(seed)
    rng.shuffle(folders)

    n = len(folders)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))

    train_folders = folders[:n_train]
    val_folders = folders[n_train:n_train + n_val]
    test_folders = folders[n_train + n_val:]
    return train_folders, val_folders, test_folders


def protocol_mono_vim(args, exts):
    seq_to_frames = discover_sequences(args.data_path, exts)
    all_folders = sorted(seq_to_frames.keys())

    train_folders, val_folders, test_folders = split_folders_disjoint(
        all_folders,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_lines = build_lines_for_folders(seq_to_frames, train_folders, args.side)
    val_lines = build_lines_for_folders(seq_to_frames, val_folders, args.side)
    test_lines = build_lines_for_folders(seq_to_frames, test_folders, args.side)
    return train_lines, val_lines, test_lines


def protocol_mono_vim_reported(args, exts):
    if not args.reported_split_file:
        raise ValueError("--reported_split_file is required when --protocol mono_vim_reported")
    with open(args.reported_split_file, "r") as f:
        table = json.load(f)

    seq_to_frames = discover_sequences(args.data_path, exts)

    train_folders = table.get("train", [])
    val_folders = table.get("val", [])
    test_folders = table.get("test", [])

    train_lines = build_lines_for_folders(seq_to_frames, train_folders, args.side)
    val_lines = build_lines_for_folders(seq_to_frames, val_folders, args.side)
    test_lines = build_lines_for_folders(seq_to_frames, test_folders, args.side)
    return train_lines, val_lines, test_lines


def summarize_split(name, lines):
    unique_folders = set()
    for line in lines:
        unique_folders.add(line.split()[0])
    print(f"{name:>5}: {len(lines):6d} frames across {len(unique_folders):4d} folders")


def main():
    parser = argparse.ArgumentParser("Generate C3VD split files (non-triplet format).")
    parser.add_argument("--data_path", type=str, required=True, help="root path of C3VD dataset")
    parser.add_argument("--output_dir", type=str, default=os.path.join("splits", "c3vd"))
    parser.add_argument(
        "--protocol",
        type=str,
        default="folder_layout",
        choices=["folder_layout", "mono_vim", "mono_vim_reported"],
        help="split protocol. This script writes only '<folder> <frame_idx> l' lines.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="used by mono_vim")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="used by mono_vim")
    parser.add_argument("--seed", type=int, default=42, help="used by mono_vim")
    parser.add_argument(
        "--reported_split_file",
        type=str,
        default=None,
        help="JSON file with keys train/val/test (used by mono_vim_reported)",
    )
    parser.add_argument("--side", type=str, default="l", help="camera token for split lines")
    parser.add_argument(
        "--extensions",
        type=str,
        default=".png,.jpg,.jpeg",
        help="comma-separated image extensions to scan",
    )
    args = parser.parse_args()

    exts = set()
    for ext in args.extensions.split(","):
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        exts.add(ext)
    if not exts:
        raise ValueError("No valid extensions provided via --extensions")

    output_dir = choose_output_dir(args.output_dir)

    if args.protocol == "folder_layout":
        train_lines, val_lines, test_lines = protocol_folder_layout(args, exts)
    elif args.protocol == "mono_vim":
        train_lines, val_lines, test_lines = protocol_mono_vim(args, exts)
    else:
        train_lines, val_lines, test_lines = protocol_mono_vim_reported(args, exts)

    write_lines(os.path.join(output_dir, "train_files.txt"), train_lines)
    write_lines(os.path.join(output_dir, "val_files.txt"), val_lines)
    write_lines(os.path.join(output_dir, "test_files.txt"), test_lines)

    print(f"Wrote split files to: {os.path.abspath(output_dir)}")
    summarize_split("train", train_lines)
    summarize_split("val", val_lines)
    summarize_split("test", test_lines)


if __name__ == "__main__":
    main()
