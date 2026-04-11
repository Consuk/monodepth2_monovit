try:
    from .kitti_dataset import KITTIRAWDataset, KITTIOdomDataset, KITTIDepthDataset
except Exception:
    # Keep non-KITTI datasets usable when optional KITTI deps are missing.
    KITTIRAWDataset = None
    KITTIOdomDataset = None
    KITTIDepthDataset = None

from .scared_dataset import SCAREDDataset, SCAREDRAWDataset
from .hamlyn_dataset import HamlynDataset

try:
    from .c3vd_dataset import C3VDDataset
except Exception:
    C3VDDataset = None

dataset_dict = {
    "endovis": SCAREDRAWDataset,
    "hamlyn": HamlynDataset,
}

if C3VDDataset is not None:
    dataset_dict["c3vd"] = C3VDDataset

if KITTIRAWDataset is not None:
    dataset_dict["kitti"] = KITTIRAWDataset
    dataset_dict["kitti_odom"] = KITTIOdomDataset
    dataset_dict["kitti_depth"] = KITTIDepthDataset
