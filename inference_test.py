import json
import os
#import sys
#sys.path.append("..")

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


#import ROI_detection2.utils.train_utils as train_utils

from ubteacher.config import add_ubteacher_config
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN

# def dataset_dict parser
# def custom Pytorch dataloader for batch inference

if __name__ == "__main__":
    print("Import successful")
    # Setup parser for cmd line arguments
    # Register dataset and metadata
    # Set inference configs
    # Load trained model, config batch size and num_workers
    # Main batch inference script