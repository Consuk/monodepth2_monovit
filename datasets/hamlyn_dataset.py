from __future__ import absolute_import, division, print_function

"""
Dataset loader for the Hamlyn endoscopic video dataset.

This implementation extends the generic :class:`MonoDataset` to load
monocular sequences from the Hamlyn dataset.  Each sample in the Hamlyn
split file is expected to be a path relative to the root of the dataset
(`self.data_path`) pointing to a single frame without file extension,
followed by an integer index and a side indicator (``l`` or ``r``).  The
frame index is preserved for compatibility with the common dataset API,
but the file name itself is used to load the image.  The dataset also
provides an intrinsic matrix scaled to the requested resolution.

Depth maps are optional.  When ``self.load_depth`` is enabled, the
``get_depth`` method attempts to locate a corresponding depth file in
``depth01`` for the left camera or ``depth02`` for the right camera.
Supported depth file extensions include ``.png``, ``.jpg``, ``.jpeg`` and
``.tiff``.  Depth values are returned as floats in metres (assuming the
dataset stores depth in millimetres).

The intrinsic matrix here is adapted from values reported in the
literature for Hamlyn endoscopic sequences.  It is normalised for
images resized to 320×256; the loader will rescale the intrinsics
automatically in :meth:`load_intrinsics` of the base class.
"""

import os
import numpy as np
import cv2
from PIL import Image as pil

# Import MonoDataset from the top-level module
from datasets.mono_dataset import MonoDataset


class HamlynDataset(MonoDataset):
    """Superclass for loading monocular frames from the Hamlyn dataset."""

    def __init__(self, *args, **kwargs):
        super(HamlynDataset, self).__init__(*args, **kwargs)
        # Intrinsic matrix normalised for 320×256 images.
        # fx = fy = 700 (64×48 sensor), cx=cy centred.  The loader rescales
        # this to the resolution set via --height and --width.
        self.K = np.array([
            [1.2270, 0.0,    0.5296, 0.0],
            [0.0,    1.2067, 0.4012, 0.0],
            [0.0,    0.0,    1.0,    0.0],
            [0.0,    0.0,    0.0,    1.0]
        ], dtype=np.float32)
        # Map side indicators to indices (unused here but provided for completeness)
        self.side_map = {"l": 2, "r": 3}

    def check_depth(self):
        # """Enable depth supervision when requested. true para cuando se entrena""" 
        return False

    def index_to_folder_and_frame_idx(self, index):
        """
        Each line in ``self.filenames`` is expected to have the form::

            <relative/path/to/frame> <frame_index> <side>

        Only ``folder`` (relative path) is needed to load the image.
        ``frame_index`` and ``side`` are preserved for API compatibility.
        """
        parts = self.filenames[index].split()
        folder = parts[0]
        frame_index = int(parts[1]) if len(parts) >= 2 else 0
        side = parts[2] if len(parts) >= 3 else None
        return folder, frame_index, side

    def get_image_path(self, folder, frame_index, side):
        """
        Compose the absolute path to an image, including the frame index.
        Example output:
        /workspace/datasets/hamlyn/Hamlyn/rectified15/rectified15/image01/0000000001.jpg
        """
        frame_str = f"{frame_index:010d}"  # 1 -> "0000000001"
        return os.path.join(self.data_path, folder, frame_str + self.img_ext)

    def get_color(self, folder, frame_index, side, do_flip):
        """Load an RGB image and optionally flip it horizontally."""
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        """
        Load the ground truth depth map (if available) for the given frame.

        It looks for depth in ``depth01`` (left camera) or ``depth02`` (right camera)
        within the same rectifiedXX/rectifiedXX directory.  Depth values are
        returned in metres; if the raw data are in millimetres the values are
        divided by 1000.0.  If no depth is found a FileNotFoundError is raised.
        """
        parts = folder.split('/')
        if len(parts) < 3:
            raise ValueError(f"Unexpected folder format: {folder}")
        base_dir = os.path.join(self.data_path, parts[0], parts[1])
        file_name = parts[-1]
        depth_dir = "depth01" if (side in ['l', 'L']) else "depth02"
        depth_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            candidate = os.path.join(base_dir, depth_dir, file_name + ext)
            if os.path.exists(candidate):
                depth_path = candidate
                break
        if depth_path is None:
            raise FileNotFoundError(f"Could not find depth map for {folder} with side {side}")
        depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        # Convert mm to m if values are large
        if depth_gt.max() > 0:
            depth_gt = depth_gt / 1000.0
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt