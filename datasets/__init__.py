from .kitti_dataset import KITTIRAWDataset, KITTIOdomDataset, KITTIDepthDataset
from .scared_dataset import SCAREDDataset
from .scared_dataset import SCAREDRAWDataset

"""
Expose dataset classes at the package level for convenience.

By importing the dataset classes here, users can do::

    from yourpackage import SCAREDRAWDataset, HamlynDataset

without needing to know the underlying file structure.  Adding
additional datasets in this file makes them available from the top level.
"""

# Import datasets so they are available at the package root.  When
# adding new datasets, import them here and update ``__all__`` below.
from .scared_dataset import SCAREDRAWDataset
from .hamlyn_dataset import HamlynDataset

dataset_dict = {
    "endovis": SCAREDRAWDataset,
    "hamlyn": HamlynDataset
}