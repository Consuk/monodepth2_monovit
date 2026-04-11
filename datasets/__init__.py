try:
    from .kitti_dataset import KITTIRAWDataset, KITTIOdomDataset, KITTIDepthDataset
except Exception:
    # Keep non-KITTI datasets usable when optional KITTI deps are missing.
    KITTIRAWDataset = None
    KITTIOdomDataset = None
    KITTIDepthDataset = None

try:
    from .scared_dataset import SCAREDDataset, SCAREDRAWDataset
except Exception:
    # SCARED depends on optional scientific stack in some environments.
    SCAREDDataset = None
    SCAREDRAWDataset = None

try:
    from .hamlyn_dataset import HamlynDataset
except Exception:
    HamlynDataset = None

try:
    from .c3vd_dataset import C3VDDataset
except Exception:
    C3VDDataset = None

dataset_dict = {}

if SCAREDRAWDataset is not None:
    dataset_dict["endovis"] = SCAREDRAWDataset

if HamlynDataset is not None:
    dataset_dict["hamlyn"] = HamlynDataset

if C3VDDataset is not None:
    dataset_dict["c3vd"] = C3VDDataset

if KITTIRAWDataset is not None:
    dataset_dict["kitti"] = KITTIRAWDataset
    dataset_dict["kitti_odom"] = KITTIOdomDataset
    dataset_dict["kitti_depth"] = KITTIDepthDataset
