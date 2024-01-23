

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
from detectron2.engine import DefaultPredictor
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.engine.trainer import UBRCNNTeacherTrainer
from ubteacher.config import add_ubteacher_config
from ubteacher.utils.train2_utils import (register_dataset,
                                          ParseUnlabeled)

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

def custom_visualizer(img, instances, gt_instances = None, cat_map = None):
    """
    Custom visualizer for UBTeacher2 inference.
    """
    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(img)
    
    # Loop over instances and draw boxes
    for i in range(len(instances)):
        if cat_map:
            cat = cat_map[str(instances[i].pred_classes.numpy()[0])]
        else:
            cat = instances[i].pred_classes.numpy()[0]
        x1, y1, x2, y2 = instances[i].pred_boxes.tensor.numpy()[0]
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, cat, fontsize=9, color='r')
        ax.text(x1, y1 + img.shape[1]/20, str(round(instances[i].scores.numpy()[0], 2)), fontsize=7, color='r')
        
        try:
            polygon = instances[i].pred_masks.numpy()[0]
            ax.imshow(polygon, alpha=0.5)
        except:
            pass
    
    try: 
        # Loop over gt instances and draw boxes
        for i in range(len(gt_instances)):
            if cat_map:
                cat = cat_map[str(gt_instances[i].gt_classes.numpy()[0])]
            else:
                cat = gt_instances[i].gt_classes.numpy()[0]
            x1, y1, x2, y2 = gt_instances[i].gt_boxes.tensor.numpy()[0]
            w = x2 - x1
            h = y2 - y1
            rect = patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='g',facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, cat, fontsize=12, color='g')
    except:
        pass
            
        return fig, ax

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
        "output_dir",
        metavar="OUTPUT_DIRECTORY",
        type=str,
        help="path to output directory for visualizations, ex: '/mnt/d/ROI_model/output'",
    )
    
    # Optional
    
    parser.add_argument(
        "-v",
        "--validation",
        action="store_true",
        help="validation mode",
        default = False
    )
    
    parser.add_argument(
        "-gt",
        "--ground_truth",
        metavar="GROUND_TRUTH_DIRECTORY",
        type=str,
        help="path to ground truth directory",
    )
    
    parser.add_argument(
        "-d",
        "--data_dir",
        metavar="DATASET_DIRECTORY",
        type=str,
        help="for inference or testing, direct path to the dataset"
    )
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
    val_mode = args.validation
    
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
        
    # If validation mode, get dataseed to perform inference on validation set
    
    if val_mode:
        with open(os.path.join(cfg.DATASEED)) as f:
            dataset_seed = json.load(f)
        imgs = []
        for dict in dataset_seed['val']:
            imgs.append(dict['file_name'])
    else:
        imgs = glob.glob(os.path.join(dataset_dir, "*.npy"))
    
    # get ground truth annos. 
    gt_instances = 
        
      
    # Get inference dicts
    inf_dicts = []
    for img_file in imgs:
        each_dict = ParseUnlabeled(img_file).get_unlabeled_coco(img_file)
        inf_dicts.append(each_dict)
    
    # Load model
    student_model = UBRCNNTeacherTrainer.build_model(cfg)
    teacher_model = UBRCNNTeacherTrainer.build_model(cfg)
    model = EnsembleTSModel(teacher_model, student_model)
    model.eval()
    used_model = model.modelStudent

    checkpointer = DetectionCheckpointer(model) 
    checkpointer.load(cfg.MODEL.WEIGHTS)   

    # Convert to batched inputs and perform inference
    
    for d in inf_dicts:
        raw_img = np.load(d[0]["file_name"])
        im = torch.from_numpy(np.transpose(raw_img, (2, 0, 1)))
        inputs = [{"image": im, "height": im.shape[1], "width": im.shape[2]}]
        with torch.no_grad():
            outputs = used_model(inputs)
            instances = outputs[0]["instances"].to("cpu")
            instances.get_fields()
            fig, ax = custom_visualizer(raw_img, instances, gt_instances, cat_map = cat_map)
            plt.show()
            plt.savefig(os.path.join(args.output_dir, d[0]["file_name"].split('/')[-1].split('.')[0] + '.png'))
            plt.close()

  
    
    
    
        
    
    
    
    
    