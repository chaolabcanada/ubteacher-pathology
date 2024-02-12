

# Imports
import os
import shutil
from typing import *
from time import perf_counter

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

def custom_visualizer(img_id, img, instances, gt_instances = None, cat_map = None):
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
    # Loop over ground truth instances and draw boxes
    if gt_instances is not None:
        for i in range(len(gt_instances[img_id])):
            if cat_map:
                gt_cat = cat_map[str(gt_instances[img_id][i]['category_id'])]
            else:
                gt_cat = gt_instances[img_id][i]['category_id']
            box = gt_instances[img_id][i]['bbox']
            x1, y1, x2, y2 = box
            gt_rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='g',facecolor='none')
            ax.add_patch(gt_rect)
            ax.text(x1, y1, gt_cat, fontsize=9, color='g')
            try:
                gt_poly = gt_instances[img_id][i]['segmentation']
                gt_poly = np.array(gt_poly).reshape((int(len(gt_poly)/2), 2))
                ax.fill(gt_poly[:, 0], gt_poly[:, 1], alpha=0.5, color='g')
            except:
                print("No segmentation for ground truth annotation.")
                pass
    return fig, ax
    
def parse_gt(gt_dir):
    """
    Parse ground truth annotations.
    """
    annos = glob.glob(os.path.join(gt_dir, "*.json"))
    gt_dict = {}
    for file in annos:
        with open(file) as f:
            ddict = json.load(f)
        img_id = file.split('/')[-1].split('.')[0]
        each_gt = []
        for i in range(len(ddict['annotations'])):
            cat = ddict['annotations'][i]['category_id']
            bbox = ddict['annotations'][i]['bbox']
            poly = ddict['annotations'][i]['segmentation'][0]
            each_gt.append({'category_id': cat, 'bbox': bbox, 'segmentation': poly})
        gt_dict.update({img_id: each_gt})
    return gt_dict
    
def qupath_coordspace(dir, tissue_crop = True):
    """
    Convert instances to QuPath coordinate space.

    Input: list of annotations to be converted from tissue space back to original space
    Output: list of annotations in original QuPath compatible space and list of annotations in tissue space
    """

    annos = glob.glob(os.path.join(dir, "*.json"))
    scaled = {}
    for file in annos:
        with open(file) as f:
            ddict = json.load(f)
        img_id = file.split('/')[-1].split('.')[0]
        if tissue_crop:
            x_offset = ddict['tissue_xyxy'][0]
            y_offset = ddict['tissue_xyxy'][1]
            x_scale = ddict['original_width'] / ddict['width']
            y_scale = ddict['original_height'] / ddict['height']
        else:
            x_offset, y_offset, x_scale, y_scale = 0, 0, 1, 1
        # Convert to tissue coord. space
        each_scaled = []
        for i in range(len(ddict['annotations'])):
            cat = ddict['annotations'][i]['category_id']
            bbox = ddict['annotations'][i]['bbox']
            poly = ddict['annotations'][i]['segmentation'][0]
            for j in range(len(bbox)):
                if j % 2 == 0:
                    bbox[j] = round(bbox[j] * x_scale + x_offset, 1)
                else:
                    bbox[j] = round(bbox[j] * y_scale + y_offset, 1)
            if len(poly) > 4:
                for k in range(len(poly)):
                    if i % 2 == 0:
                        poly[k] = round(poly[k] * x_scale + x_offset, 1)
                    else:
                        poly[k] = round(poly[k] * y_scale + y_offset, 1)
            else:
                poly = None
            each_scaled.append({'category_id': cat, 'bbox': bbox, 'segmentation': poly})
        scaled.update({img_id: each_scaled})
    return scaled

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
        "-t",
        "--tissue_cropping",
        action="store_true",
        default = False,
        help="use a model which crops out tissue before inference",
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
    gt_dir = args.ground_truth
    tissue_cropping = args.tissue_cropping
    
    cfg = get_cfg()
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
        gt_qupath = qupath_coordspace(gt_dir, tissue_cropping)
        gt_tissue = parse_gt(gt_dir)
    
        with open(os.path.join(cfg.DATASEED)) as f:
            dataset_seed = json.load(f)
        imgs = []
        
        try: # with individual dicts
            for dict in dataset_seed['val']:
                imgs.append(dict['file_name'])
        except: # with list of files
            imgs = dataset_seed['val']['images']
    else:
        imgs = glob.glob(dataset_dir + "/*.npy")
       
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

    global_tick = perf_counter()
    # Convert to batched inputs and perform inference
    for d in inf_dicts:
        raw_img = np.load(d[0]["file_name"])
        im = torch.from_numpy(np.transpose(raw_img, (2, 0, 1)))
        img_id = d[0]['file_name'].split('/')[-1].split('.')[0]
        inputs = [{"image": im, "height": im.shape[1], "width": im.shape[2]}]
        with torch.no_grad():
            outputs = used_model(inputs)
            instances = outputs[0]["instances"].to("cpu")
            instances.get_fields()
            if val_mode:
                fig, ax = custom_visualizer(img_id, raw_img, instances, gt_tissue, cat_map = cat_map)
            else:
                fig, ax = custom_visualizer(img_id, raw_img, instances, cat_map = cat_map)
            #plt.show()
            plt.savefig(os.path.join(args.output_dir, img_id + '.png'))
            plt.close()
    print(f"Total time: {perf_counter() - global_tick} for {len(inf_dicts)} images")
    print(f"avg. {round((perf_counter() - global_tick)/len(inf_dicts), 2)}s per image)")
    
    # Convert outputs to QuPath coordinate space
      
    

  
    
    
    
        
    
    
    
    
    