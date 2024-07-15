# TODO Make an eval script to output COCO style APs and ARs

# Imports
import argparse
import glob
import json
import os
from pathlib import Path
import shutil
import sys
# Hacky way to resolve project paths
sys.path.append(str(Path(os.getcwd()).parents[0]))
sys.path.append(str(Path(os.getcwd()).parents[1]))
from time import perf_counter
from typing import *

import cv2
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.engine.trainer import UBRCNNTeacherTrainer
from ubteacher.config import add_ubteacher_config
from ubteacher.utils.train2_utils import (register_dataset, channel_last,
                                          ParseUnlabeled, merge_bboxes)

## TODO Cat map support for merged boxes

def custom_visualizer(img_id, 
                      img, 
                      instances, 
                      gt_instances = None, 
                      merged_bboxes = False, 
                      cat_map = None):
    """
    Custom visualizer for UBTeacher2 inference.
    """
    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(img)
    # Loop over instances and draw boxes
    if merged_bboxes:
        for i in range(len(instances)):
            if instances[i]['score'] > threshold:
                x1, y1, x2, y2 = instances[i]['bbox']
                w = x2 - x1
                h = y2 - y1
                rect = patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
    else:
        for i in range(len(instances)):
            if cat_map:
                cat = cat_map[str(instances[i].pred_classes.numpy()[0])]
            else:
                cat = instances[i]['category_id']
            score = float(instances[i]['score'])
            if score > threshold:
                x1, y1, x2, y2 = instances[i]['bbox']
                w = x2 - x1
                h = y2 - y1
                rect = patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, cat, fontsize=9, color='r')
                ax.text(x1, y1 + img.shape[1]/20, str(round(instances[i]['score'], 2)), fontsize=7, color='r')   
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

    
def qupath_coordspace(instance_dicts, wsi_path, img, tissue_crop = True):
    """
    Convert preds to QuPath coordinate space.
    """ 
    ## TODO: Support tisue crop
    # TODO Use one package to load wsi and get properties, don't mix openslide and tifffile
    # get wsi dimensions
    slide = openslide.OpenSlide(wsi_path)
    mppx = float(slide.properties['openslide.mpp-x'])
    mppy = float(slide.properties['openslide.mpp-y'])
        
    with tf.TiffFile(wsi_path) as slide:
        print(slide.pages[0].description)
        try:
            base_dim = slide.series[0].levels[0].shape
            base_dim = channel_last(base_dim)
        except:
            # load image
            image = tf.imread(wsi_path)
            base_dim = channel_last(image.shape)
    if not tissue_crop:
        scale_x = base_dim[0] / img.shape[0]
        scale_y = base_dim[1] / img.shape[1]
    for i in instance_dicts:
        x1, y1, x2, y2 = i['bbox']
        x1 = x1 * scale_x
        x2 = x2 * scale_x
        y1 = y1 * scale_y
        y2 = y2 * scale_y
        i['bbox'] = [x1, y1, x2, y2]
        i['width_microns'] = (x2 - x1) * mppx
        i['height_microns'] = (y2 - y1) * mppy
    return instance_dicts           

def save_annos(qupath_dicts, threshold):
    """
    Save annotations to a json file
    """
    ## TODO: Cat map support
    out_dicts = []
    for i in range(len(qupath_dicts)):
        score = float(qupath_dicts[i]['score'])
        box = qupath_dicts[i]['bbox']
        try:
            cat = qupath_dicts[i]['category_id']
        except:
            cat = 0
        if score < threshold:
            continue
        out_dicts.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                                [box[0], box[1]],
                                [box[2], box[1]],
                                [box[2], box[3]],
                                [box[0], box[3]],
                                [box[0], box[1]]
                                ]]
                        },
            "properties": {
                "objectType": "annotation",
                "name": str(f"{i['width_microns']}x{i['height_microns']}"),
                "color": [0, 255, 0],
                "classification": {
                    "name": str(cat),
                }
                    }
            })
    qupath_out = os.path.join(args.output_dir, 'qupath_predictions')
    if not os.path.exists(qupath_out):
        os.makedirs(qupath_out)
    with open(os.path.join(qupath_out, img_id + '.json'), 'w') as f:
        json.dump(out_dicts, f, indent=4)
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict boxes or masks using a trained ubteacher-v2 model. \
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
    wsi_dir = args.wsi_dir
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
    
    # Assemble student-teacher model, load checkpoint
    student_model = UBRCNNTeacherTrainer.build_model(cfg)
    teacher_model = UBRCNNTeacherTrainer.build_model(cfg)
    model = EnsembleTSModel(teacher_model, student_model)
    model.eval()
    inference_model = model.modelTeacher # Use the teacher model for inference
    checkpointer = DetectionCheckpointer(model) 
    checkpointer.load(cfg.MODEL.WEIGHTS)   

    # Perform inference
    global_tick = perf_counter()
    filter_count = 0
    all_ratios = {}
    for d in inf_dicts:
        tick = perf_counter()
        raw_img = np.load(d[0]["file_name"])
        wsi_path = os.path.join(wsi_dir, d[0]["file_name"].split('/')[-1].split('.')[0] + '.svs')
        im = torch.from_numpy(np.transpose(raw_img, (2, 0, 1)))
        img_id = d[0]['file_name'].split('/')[-1].split('.')[0]
        inputs = [{"image": im, "height": im.shape[1], "width": im.shape[2]}]
        
        with torch.no_grad():
            outputs = inference_model(inputs)
            instances = outputs[0]["instances"].to("cpu")
            instance_dicts = []
            for i in range(len(instances)):
                instance_dicts.append({'category_id': instances[i].pred_classes.numpy()[0], 
                                       'bbox': instances[i].pred_boxes.tensor.numpy()[0].tolist(),
                                       'score': instances[i].scores.numpy()[0]})
            final_anno = instance_dicts
            #merged = merge_bboxes(instance_dicts, threshold)
            #final_anno, filter_count, ratios = nuc_seg(raw_img, instance_dicts,
            #                                         wsi_path, filter_count)
            #all_ratios.update({img_id: ratios})
            #merged = merge_bboxes(filtered, threshold)
            if val_mode:
                fig, ax = custom_visualizer(
                    img_id, raw_img, final_anno, gt_tissue, 
                    merged_bboxes = False, cat_map = cat_map)
            else:
                fig, ax = custom_visualizer(img_id, raw_img, final_anno, 
                                            merged_bboxes = False, cat_map = cat_map)
            #plt.show()
            plt.savefig(os.path.join(args.output_dir, img_id + '.png'))
            plt.close()
            qupath_dicts = qupath_coordspace(final_anno, wsi_path, raw_img, 
                                             tissue_crop = tissue_cropping)
            save_annos(qupath_dicts, threshold)
            toc = perf_counter()
            print(f"Processed {img_id} in {round(toc - tick, 2)}s")
            print(f"Filtered {filter_count} instances.")
    print(f"Total time: {perf_counter() - global_tick} for {len(inf_dicts)} images")
    print(f"Average time: {round((perf_counter() - global_tick)/len(inf_dicts), 2)}s per image)")
   