import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SyntheWorldDataset(CustomDataset):
    """SyntheWorld dataset.

    This class defines the dataset structure for SyntheWorld, adapting
    from a typical semantic segmentation dataset class.
    """
    # Note: If your SyntheWorld labels use 0 as a valid class, you might need
    # to adjust `reduce_zero_label` in the __init__ method or handle it
    # appropriately in your data pipeline.
    CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest',
               'agricultural')

    # Define a color palette for visualization.
    # The number of colors must match the number of CLASSES.
    # These are example colors; you can customize them.
    PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
               [159, 129, 183], [0, 255, 0], [255, 195, 128]]

    def __init__(self, **kwargs):
        super(SyntheWorldDataset, self).__init__(
            img_suffix='.png',          # Assuming image files are .png
            seg_map_suffix='.png',      # Assuming segmentation maps (labels) are .png
            reduce_zero_label=True,    # Set to False if 0 is a valid class ID in SyntheWorld labels
                                        # Set to True if 0 is an ignore index (like in LoveDA)
            **kwargs)
