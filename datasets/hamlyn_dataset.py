from __future__ import absolute_import, division, print_function

import os
import numpy as np
from PIL import Image as pil

from datasets.mono_dataset import MonoDataset

class HamlynDataset(MonoDataset):
    """
    Hamlyn dataset loader for Monodepth2 with:
      - ENDO-DAC-style folder mapping: rectifiedXX -> rectifiedXX/rectifiedXX
      - Per-sequence intrinsics from intrinsics.txt (3x4 or 3x3)

    Typical on-disk structure:
      Hamlyn/
        rectified08/
          rectified08/
            intrinsics.txt
            image01/000000####.jpg
            image02/000000####.jpg
            ...

    Each line in the split files is expected to contain:
      <sequence_folder> <frame_index> <side>
    e.g. "rectified08 1749 l"
    """
    def __init__(self, *args, **kwargs):
        # Accept strict-neighbor args (used by --hamlyn_strict_neighbors)
        strict_neighbors = kwargs.pop("strict_neighbors", None)
        if strict_neighbors is None:
            strict_neighbors = kwargs.pop("hamlyn_strict_neighbors", False)
        neighbor_search_max = kwargs.pop("neighbor_search_max", 10)

        # Set BEFORE MonoDataset.__init__ so its getattr(self, ...) picks them up.
        self.strict_neighbors = bool(strict_neighbors)
        self.neighbor_search_max = int(neighbor_search_max)

        super(HamlynDataset, self).__init__(*args, **kwargs)
        self._K_cache = {}           # Cache normalized intrinsics per sequence (4x4)
        self._actual_seq_cache = {}  # Cache mapping folder name to actual sequence root path

        # Map side letters to camera subfolder names
        self._side_to_cam = {"l": "image01", "r": "image02", "L": "image01", "R": "image02"}

        # These attributes can be set externally (e.g., in Trainer)
        self.strict_neighbors = getattr(self, "strict_neighbors", False)
        self.neighbor_search_max = int(getattr(self, "neighbor_search_max", 10))

    def check_depth(self):
        return False

    def index_to_folder_and_frame_idx(self, index):
        parts = self.filenames[index].split()
        folder = parts[0]
        frame_index = int(parts[1]) if len(parts) >= 2 else 0
        side = parts[2] if len(parts) >= 3 else "l"
        return folder, frame_index, side

    # -------------------------------------------------------------------------
    # ENDO-DAC style folder resolution
    # -------------------------------------------------------------------------
    def _resolve_actual_sequence_root(self, folder):
        """
        If 'folder' is 'rectifiedXX', map to 'rectifiedXX/rectifiedXX' if it exists.
        If folder already contains 'rectifiedXX/rectifiedXX', use it as-is.
        Caches results for performance.
        """
        norm = folder.replace("\\", "/").strip("/")
        if norm in self._actual_seq_cache:
            return self._actual_seq_cache[norm]
        cand1 = norm
        base = norm.split("/")[0]
        cand2 = os.path.join(base, base).replace("\\", "/")
        def looks_valid(seq_rel):
            intr = os.path.join(self.data_path, seq_rel, "intrinsics.txt")
            img1 = os.path.join(self.data_path, seq_rel, "image01")
            img2 = os.path.join(self.data_path, seq_rel, "image02")
            return os.path.exists(intr) or os.path.isdir(img1) or os.path.isdir(img2)
        if looks_valid(cand1):
            actual = cand1
        elif looks_valid(cand2):
            actual = cand2
        else:
            actual = cand1  # default to original if neither candidate exists
        self._actual_seq_cache[norm] = actual
        return actual

    def _resolve_sequence_and_cam_dir(self, folder, side):
        """
        Determine the sequence directory and camera subfolder for a given folder token and side.
        Handles cases where 'folder' is just the sequence or already includes camera.
        Returns (seq_dir_rel, cam_dir, folder_rel).
        """
        norm = folder.replace("\\", "/").rstrip("/")
        # If folder includes camera subfolder:
        if norm.endswith("/image01") or norm.endswith("/image02"):
            seq_dir_rel = os.path.dirname(norm)
            cam_dir = os.path.basename(norm)
            seq_dir_rel = self._resolve_actual_sequence_root(seq_dir_rel)
            folder_rel = os.path.join(seq_dir_rel, cam_dir).replace("\\", "/")
            return seq_dir_rel, cam_dir, folder_rel
        cam_dir = self._side_to_cam.get(side, "image01")
        seq_dir_rel = self._resolve_actual_sequence_root(norm)
        folder_rel = os.path.join(seq_dir_rel, cam_dir).replace("\\", "/")
        return seq_dir_rel, cam_dir, folder_rel

    # -------------------------------------------------------------------------
    # Image loading
    # -------------------------------------------------------------------------
    def get_image_path(self, folder, frame_index, side):
        _, _, folder_rel = self._resolve_sequence_and_cam_dir(folder, side)
        frame_str = f"{frame_index:010d}"
        return os.path.join(self.data_path, folder_rel, frame_str + self.img_ext)

    def get_color(self, folder, frame_index, side, do_flip):
        path = self.get_image_path(folder, frame_index, side)
        color = self.loader(path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    # -------------------------------------------------------------------------
    # Intrinsics handling
    # -------------------------------------------------------------------------
    def _read_intrinsics_txt(self, intr_path):
        rows = []
        with open(intr_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = [float(x) for x in line.split()]
                rows.append(vals)
        if len(rows) < 3:
            raise ValueError(f"Invalid intrinsics file (need at least 3 rows): {intr_path}")
        M = np.zeros((3, 4), dtype=np.float32)
        for r in range(3):
            row = rows[r]
            if len(row) >= 4:
                M[r, :] = row[:4]
            elif len(row) == 3:
                M[r, :3] = row
            else:
                raise ValueError(f"Row {r} has too few columns in {intr_path}: {row}")
        fx, fy = float(M[0, 0]), float(M[1, 1])
        cx, cy = float(M[0, 2]), float(M[1, 2])
        return fx, fy, cx, cy

    def _get_native_image_size(self, folder, frame_index):
        # Determine original image dimensions (assuming both stereo images same size)
        path = self.get_image_path(folder, frame_index, "l")
        with pil.open(path) as img:
            w, h = img.size
        return w, h

    def load_intrinsics(self, folder, frame_index):
        """
        Returns the normalized 4x4 intrinsics matrix for the sequence of the given frame.
        (MonoDataset will then scale this matrix for each scale.)
        """
        seq_dir_rel = self._resolve_actual_sequence_root(folder)
        if seq_dir_rel in self._K_cache:
            return self._K_cache[seq_dir_rel].copy()
        intr_path = os.path.join(self.data_path, seq_dir_rel, "intrinsics.txt")
        if not os.path.exists(intr_path):
            raise FileNotFoundError(f"Could not find intrinsics.txt in {seq_dir_rel}")
        fx, fy, cx, cy = self._read_intrinsics_txt(intr_path)
        # Normalize intrinsics if values appear to be pixel units
        pixel_like = (fx > 10.0) or (fy > 10.0) or (cx > 10.0) or (cy > 10.0)
        if pixel_like:
            w0, h0 = self._get_native_image_size(folder, frame_index)
            fx_n = fx / float(w0); fy_n = fy / float(h0)
            cx_n = cx / float(w0); cy_n = cy / float(h0)
        else:
            fx_n, fy_n, cx_n, cy_n = fx, fy, cx, cy
        K_norm = np.array([
            [fx_n, 0.0,  cx_n, 0.0],
            [0.0,  fy_n, cy_n, 0.0],
            [0.0,  0.0,  1.0,  0.0],
            [0.0,  0.0,  0.0,  1.0]
        ], dtype=np.float32)
        self._K_cache[seq_dir_rel] = K_norm.copy()
        return K_norm.copy()

    def get_depth(self, folder, frame_index, side, do_flip):
        # No ground-truth depth available for Hamlyn in self-supervised training
        raise NotImplementedError("Self-supervised training: depth ground truth not used for Hamlyn dataset")
