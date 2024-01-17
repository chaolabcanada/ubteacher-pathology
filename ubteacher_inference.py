

# Imports
import os
import shutil
from typing import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tifffile as tf
import torch
import json
import glob
import argparse


import sys
from pathlib import Path

# Hacky way to resolve project paths
sys.path.append(str(Path(os.getcwd()).parents[0]))
sys.path.append(str(Path(os.getcwd()).parents[1]))

from PIL import Image
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN # required to load model
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.engine.trainer import UBRCNNTeacherTrainer
from ubteacher.config import add_ubteacher_config
from ubteacher.utils.train2_utils import (register_dataset,
                                          ParseUnlabeled)

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

if __name__ == "__main__":
    # Necessary
    parser = argparse.ArgumentParser(
        description="Predict boxes or masks using a trained UBTeacher2 model. \
                    By default predicts boxes; \
                    use '--mask' to predict masks instead."
    )
    parser.add_argument(
        "model",
        metavar="MODEL_PATH",
        type=str,
        help="path to the model file, ex: '/mnt/d/ROI_model/model.pth'",
    )
    parser.add_argument(
        "config",
        metavar="CONFIG_PATH",
        type=str,
        help="path to the training config file, ex: '/mnt/d/ROI_model/config.yaml'",
    )
    
    parser.add_argument(
        "data_dir",
        metavar="DATASET_DIRECTORY",
        type=str,
        help="for inference or testing, direct path to the dataset; \
                for validation, parent directory of validation data "
    )
    
    parser.add_argument(
        "output_dir",
        metavar="OUTPUT_DIRECTORY",
        type=str,
        help="path to output directory for visualizations, ex: '/mnt/d/ROI_model/output'",
    )
    
    # Optional
    parser.add_argument(
        "-cm",
        "--category_map",        
        metavar="CATEGORY_MAP_PATH",
        type=str,
        help="path to categorical mapping, ex: '/mnt/d/ROI_model/category_map.json",
    )    
    parser.add_argument(
        "-m",
        "--mask",
        action="store_true",
        help="predict masks as well as boxes",
        default = False
    )
    parser.add_argument(
        "-th",
        "--threshold",
        metavar='DETECTION_THRESHOLD',
        type=float,
        nargs="?",
        default=0.7,
        help="set detection threshold; higher=more stringent. default=0.7",
    )
    parser.add_argument(
        "-o",
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Modify any config option using command-line 'KEY VALUE' pairs. \
                            Ex: '--opts MODEL.ROI_HEADS.NMS_THRESH_TEST 0.5'",
    )
    
    args = parser.parse_args()
    model_path = args.model
    dataset_dir = args.data_dir
    config_path = args.config
    mask = args.mask
    threshold = args.threshold
    
    # Load config
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    
    # Create output folder
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load cat map
    if args.category_map:
        with open(args.category_map) as json_file:
            cat_map = json.load(json_file)
    else:
        cat_map = None
        print("No category map specified. Using numerical classes.")
        
    # Register inference dataset
    inf_dicts = []
    for img_file in glob.glob(os.path.join(dataset_dir, "*.npy")):
        each_dict = ParseUnlabeled(img_file).get_unlabeled_coco(img_file)
        inf_dicts.append(each_dict)
    
    # Load model
    
    student_model = UBRCNNTeacherTrainer.build_model(cfg)
    teacher_model = UBRCNNTeacherTrainer.build_model(cfg)
    model = EnsembleTSModel(teacher_model, student_model)
    model.eval()
    
    checkpointer = DetectionCheckpointer(model) 
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # Convert to batched inputs and perform inference
    
    for d in inf_dicts:
        raw_img = np.load(d[0]["file_name"])
        im = torch.from_numpy(np.transpose(raw_img, (2, 0, 1)))
        inputs = [{"image": im, "height": im.shape[1], "width": im.shape[2]}]
        with torch.no_grad():
            outputs = model.modelTeacher(inputs)
            instances = outputs[0]["instances"].to("cpu")
            instances.get_fields()
            v = Visualizer(raw_img, metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=2.0) # check how metadata list comprehension works
            v = v.draw_instance_predictions(instances)
            plt.imshow(v.get_image())
            plt.show()
            plt.savefig(os.path.join(args.output_dir, d[0]["file_name"].split("/")[-1].split(".")[0] + ".png"))
            plt.close()
    
    
    
    
        
    
    
    
    
    