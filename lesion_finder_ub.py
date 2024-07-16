"""
Use a UBTeacher model to perform inference, where suspected lesions are identified as regions of interest (ROIs) and saved as images.
Can be used on tissue numpy arrays or directly on whole-slide images plus a QuPath-compatible json to reference the tissue coordinates.

TODO 
- test the script on set 3

@Version: 0.1
@Author: Chao Lab, SRI
@Contact: jesse.chao@sri.utoronto.ca
"""

  ## function to read the json from tissue finder outputs (QP format)
    ## DONT MERGE!!!! THE BOXESA !!!
    ## function to write the json w.r.t. the entire image (i.e. add the tissue offsets before rescaling to base dim)
    ## lesionfinder ub model stuff should be the same etc
import argparse
import copy
from functools import partial
import json
import os
from pathlib import Path
import shutil
import sys
import time

import tifffile as tf
import torch
import torch.multiprocessing as mp
import torch.utils.data as torchdata
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import MapDataset
from detectron2.data import transforms as T
from detectron2.data.samplers import InferenceSampler
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultPredictor
from ubteacher.config import add_ubteacher_config
from ubteacher.engine.trainer import UBRCNNTeacherTrainer
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

import utils.train_utils as train_utils
import utils.gradcam as gradcam



def handle_existing_output_folder(output_dir: str) -> str:
    """
    Handle existing output folder by creating a new version if needed.

    Args
        output_dir (str): Existing output directory.
    Returns
        str: Updated output directory.
    """
    if os.path.exists(output_dir):
        while True:
            overwrite_ans = input("The output folder already exists! Would you like to overwrite it (y/n)?")
            if overwrite_ans.lower() == 'n':
                version_number = 2
                while os.path.exists(output_dir + f"_v{version_number}"):
                    version_number += 1
                output_dir = output_dir + f"_v{version_number}"
                break
            elif overwrite_ans.lower() == 'y':
                print("    the existing output folder will be removed")
                shutil.rmtree(output_dir)
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    return output_dir

class DatasetHelper:
    def __init__(self, cfg: CfgNode) -> None:
            self.max_dim = cfg.INPUT.MAX_SIZE_TRAIN  # The max num pixels of the longest edge
            self.num_workers = cfg.DATALOADER.NUM_WORKERS
            self.batch_size = cfg.DATALOADER.BATCH_SIZE
            self.compatible_formats = (
                "tif",
                "tiff",
                "stk",
                "lsm",
                "sgi",
                "gel",
                "svs",
                "scn",
                "sis",
                "bif",
                "zif",
            )

    def get_img_data(self, img_dir: str) -> list:
        dataset = []
        for file in os.scandir(img_dir):
            if not file.name.startswith('.') and file.name.endswith(self.compatible_formats):
                entry = {}
                entry['file_name'] = file.path
                entry['image_id'] = file.name
                dataset.append(entry)
        return dataset
    
    def get_tissue_data(self, img_dir: str, json_dir: str) -> dict:
        tissue_dataset = []
        for file in os.scandir(img_dir):
            if not file.name.startswith('.') and file.name.endswith(self.compatible_formats):
                corr_json = os.path.join(json_dir, f'pred_{file.name.split('.')[0]}.json')
                if not os.path.exists(corr_json):
                    print(f"Skipping {file.name} as corresponding json file does not exist.")
                    continue
                with open(corr_json, "r") as f:
                    tissue_data = json.load(f)
            for tissue in tissue_data: # for each tissue make an entry
                entry = {}
                entry['file_name'] = file.path
                entry['image_id'] = file.name
                x0 = tissue['coordinates'][0][0]
                y0 = tissue['coordinates'][0][1]
                x1 = tissue['coordinates'][2][0]
                y1 = tissue['coordinates'][2][1]
                entry['tissue'] = [x0, y0, x1, y1]
            tissue_dataset.append(entry)
        return tissue_dataset
    
    ## rewrite this with getting only the tissue regions from tissues
    def lf_dataset_mapper(self, dataset_dict: dict) -> dict:
        entry = copy.deepcopy(dataset_dict)
        try:
            helper = train_utils.TrainUtil(self.max_dim)
            x0, y0, x1, y1 = entry['tissue']
            cropped = helper.crop_image_to_dim(entry['file_name'], [x0, y0, x1, y1], self.max_dim)
            entry['image'] = torch.from_numpy(cropped.transpose(2, 0, 1).copy())
            entry['height'] = cropped.shape[0]
            entry['width'] = cropped.shape[1]
            entry['base_height'] = y1 - y0
            entry['base_width'] = x1 - x0
            entry['src_im_height'] = cropped.shape[0]
            entry['src_im_width'] = cropped.shape[1]
        except:
            print(f"TiffFile processing error, skipping {entry['file_name']}")
        return entry
    
    def build_batch_dataloader(self, dataset):
        """
        Build a batched dataloader
        
        Args
            dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset
        Returns
            iterable[list]. Length of each list is the batch size. Each element in the
                list comes from the dataset
        """
        dataset = MapDataset(dataset, self.lf_dataset_mapper)
        return torchdata.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=InferenceSampler(len(dataset)),
            num_workers=self.num_workers,
            collate_fn=self.trivial_batch_collator
        )
    
    def trivial_batch_collator(self, batch):
        """
        A batch collator that does nothing.
        """
        return batch
    
if __name__ == "__main__":
    # ---------------------------------------
    # Setup commandline arguments
    # ---------------------------------------
    parser = argparse.ArgumentParser(
        description="Predict Lymph/Non-lymph ROIs. \
                    By default performs inference on unseen whole slide images. \
                    "
    )
    parser.add_argument(
        "model",
        metavar="MODEL_PATH",
        type=str,
        help="path to the model file, ex: '/mnt/d/ROI_model/model.pth'",
    )
    parser.add_argument(
        "data_dir",
        metavar="DATASET_DIRECTORY",
        type=str,
        help="direct path to dataset containing image files"
    )
    parser.add_argument(
        "tissue_json",
        metavar="TISSUE_JSON",
        type=str,
        help="path to the QuPath-compatible json file containing tissue coordinates"
    )
    parser.add_argument(
        '-s',
        '--size',
        metavar='IMAGE_SIZE',
        type=int,
        nargs="?",
        help='change inference resolution; default is the same as train time resolution'
    ) 
    parser.add_argument(
        "-gc",
        "--gradcam",
        action="store_true",
        help="outputs GradCAM visualizations",
    )
    parser.add_argument(
        "-th",
        "--threshold",
        metavar='DETECTION_THRESHOLD',
        type=float,
        nargs="?",
        default=0.8,
        help="set detection threshold; higher=more stringent; default=0.9",
    )
    parser.add_argument(
    "--num_workers",
    metavar='NUM_WORKERS',
    type=int,
    nargs="?",
    default=1,
    help="number of CPU workers for dataloading;  \
            also sets the number of multiprocessing workers; \
            default=1",
    )
    parser.add_argument(
    "--batch_size",
    metavar='BATCH_SIZE',
    type=int,
    nargs="?",
    default=1,
    help="number of batch size for dataloading; \
            default=1",
    )
    parser.add_argument(
        "-o",
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Modify any config option using command-line 'KEY VALUE' pairs. \
                Ex: '--opts MODEL.ROI_HEADS.NMS_THRESH_TEST 0.5'",
    )
    # Parse commandline arguments
    args = parser.parse_args()
    model_path = args.model
    model_path_parent = str(Path(model_path).parent)
    dataset_dir = str(args.data_dir[:-1]) if args.data_dir.endswith('/') else str(args.data_dir)
    json_dir = str(args.tissue_json[:-1]) if args.tissue_json.endswith('/') else str(args.tissue_json)
    reg_name = os.path.basename(dataset_dir)
    thresh = args.threshold
    make_gradcam = args.gradcam
    num_workers, batch_size = (1, 1) if make_gradcam else (args.num_workers, args.batch_size) # Set to single inference if GradCAM is requested

    # ---------------------------------------
    # Configure inference environment
    # ---------------------------------------
    # Set detection configs
    print("\nSetting detection configs...")
    print("Loading model...")
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(os.path.join(model_path_parent, "config.yaml"))
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TEST = reg_name
    cfg.DATASEED = "" # for training purposes only
    cfg.DETECTION_MODE = ""
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.DATALOADER.BATCH_SIZE = batch_size
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    if args.size is not None:
        img_size = args.size
        print(f"Changing inference resolution to {img_size}...")
    else:
        img_size = cfg.INPUT.MAX_SIZE_TRAIN
        cfg.INPUT.MAX_SIZE_TEST = img_size
    try:
        cat_map = cfg.CATEGORICAL_MAP[0]
    except IndexError or AttributeError:
        print("'CATEGORICAL_MAP' was not loaded from config! Please check!")
    try:
        detection_mode = cfg.DETECTION_MODE
    except AttributeError:
        print.error(
            "'DETECTION_MODE' is missing from the config file! Please check!"
        )
    if cfg.DETECTION_MODE == "single":
        print("    the loaded model was trained for single ROI detection")
    else:
        print(f"    the loaded model was trained to detect ROIs as {list(cat_map.keys())}")
    
    print("Configuring output directory...")
    output_dir = os.path.join(dataset_dir, f"tissue_finder_{img_size}")
    output_dir = handle_existing_output_folder(output_dir)
    print(f"    outputs from this session will be saved to '{output_dir}'")
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)