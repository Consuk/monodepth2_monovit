import os
import random
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from PIL import Image
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms

cv2.setNumThreads(0)

def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")

class MonoDataset(data.Dataset):
    """Superclass for monocular dataset loaders"""
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext=".jpg"):
        super(MonoDataset, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales

        self.interp = Image.Resampling.LANCZOS
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # Hamlyn neighbor search parameters (set via dataset subclass if applicable)
        self.strict_neighbors = getattr(self, "strict_neighbors", False)
        self.neighbor_search_max = int(getattr(self, "neighbor_search_max", 10))

        # Setup color jitter augmentation
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            # Older torchvision versions use scalar values
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        # Precompute resized image transforms for each scale
        self.resize = {}
        self.resize_mask = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=self.interp)
            self.resize_mask[i] = transforms.Resize(
                (self.height // s, self.width // s),
                interpolation=Image.Resampling.NEAREST,
            )

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize color images to required scales and apply data augmentation."""
        for k in list(inputs):
            if isinstance(k, tuple) and len(k) == 3 and k[0] == "color":
                n, img_id, intr = k
                # resize original (scale -1) to all scales
                for i in range(self.num_scales):
                    inputs[(n, img_id, i)] = self.resize[i](inputs[(n, img_id, i - 1)])
            elif isinstance(k, tuple) and len(k) == 3 and k[0] == "valid_mask":
                n, img_id, intr = k
                for i in range(self.num_scales):
                    inputs[(n, img_id, i)] = self.resize_mask[i](inputs[(n, img_id, i - 1)])
        for k in list(inputs):
            f = inputs[k]
            if isinstance(k, tuple) and len(k) == 3 and k[0] == "color":
                n, img_id, i = k
                inputs[(n, img_id, i)] = self.to_tensor(f)
                if inputs[(n, img_id, i)].sum() == 0:
                    inputs[(n + "_aug", img_id, i)] = inputs[(n, img_id, i)]
                else:
                    inputs[(n + "_aug", img_id, i)] = self.to_tensor(color_aug(f))
            elif isinstance(k, tuple) and len(k) == 3 and k[0] == "valid_mask":
                n, img_id, i = k
                m = self.to_tensor(f)
                inputs[(n, img_id, i)] = (m > 0.5).float()

    def __len__(self):
        return len(self.filenames)

    def load_intrinsics(self, folder, frame_index):
        # Default behavior: return constant intrinsics matrix (to be overridden by dataset subclasses)
        return self.K.copy()

    def _hamlyn_find_nearest_existing_frame(self, folder, frame_index, side):
        """
        For Hamlyn: if a neighbor frame is missing, find the nearest frame index that exists on disk.
        """
        # Direct attempt
        p0 = self.get_image_path(folder, frame_index, side)
        if os.path.exists(p0):
            return frame_index
        # Search outward
        for d in range(1, self.neighbor_search_max + 1):
            for cand in (frame_index + d, frame_index - d):
                p = self.get_image_path(folder, cand, side)
                if os.path.exists(p):
                    return cand
        return None

    def __getitem__(self, index):
        inputs = {}
        do_flip = self.is_train and random.random() > 0.5
        do_color_aug = self.is_train and random.random() > 0.5

        folder, frame_index, side = self.index_to_folder_and_frame_idx(index)

        # Load images for each required frame
        for i in self.frame_idxs:
            if i == "s":
                # stereo: load opposite side frame at same index
                other_side = {"l": "r", "r": "l"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                target_index = frame_index + i
                try:
                    inputs[("color", i, -1)] = self.get_color(folder, target_index, side, do_flip)
                except FileNotFoundError as e:
                    # Hamlyn neighbor snapping if enabled
                    is_hamlyn = type(self).__name__ == "HamlynDataset"
                    if is_hamlyn and self.strict_neighbors and i != 0:
                        nearest = self._hamlyn_find_nearest_existing_frame(folder, target_index, side)
                        if nearest is not None:
                            inputs[("color", i, -1)] = self.get_color(folder, nearest, side, do_flip)
                        else:
                            inputs[("color", i, -1)] = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
                    else:
                        if i != 0:
                            inputs[("color", i, -1)] = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
                        else:
                            raise FileNotFoundError(f"Cannot find frame - check data path and split files: {e}")

        # Prepare intrinsics for each scale
        for scale in range(self.num_scales):
            K = self.load_intrinsics(folder, frame_index)
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if getattr(self, "load_valid_mask", False):
            inputs[("valid_mask", 0, -1)] = self.get_valid_mask(folder, frame_index, side, do_flip)

        # Apply data augmentation (if any)
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        self.preprocess(inputs, color_aug)

        # Remove original PIL images to save memory
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
        if ("valid_mask", 0, -1) in inputs:
            del inputs[("valid_mask", 0, -1)]
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_valid_mask(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
