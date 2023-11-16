"""Validate/ test a ROI detection model or use it for inference, where lesions (neoplastic
or non-neoplastic) are detected as ROIs in whole slide images.

@Version: 0.7.0
@Author: Jesse Chao, PhD
@Contact: jesse.chao@sri.utoronto.ca
"""

import os
import shutil
import json
import sys
#import copy
import argparse
#import logging
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 6})
import matplotlib
matplotlib.use('Agg')

# Hacky way to resolve project paths
sys.path.append(str(Path(os.getcwd()).parents[0]))
sys.path.append(str(Path(os.getcwd()).parents[1]))

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import Boxes

from ubteacher.utils.ROI_utils import TrainHelper
import ubteacher.utils.eval_utils as eval_utils

from ubteacher.config import add_ubteacher_config
from ubteacher.engine.trainer import UBRCNNTeacherTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict ROIs using a trained model. \
                    By default will perform validation; \
                    requires traintime 'train-val-split.json' to be in the same dir as the model. \
                    Use '-i' for inference or '-t' for testing."
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
        help="for inference or testing, direct path to the dataset; \
                for validation, parent directory of validation data "
    )
    parser.add_argument(
        "-i",
        "--inference",
        action="store_true",
        help="performs inference; \
                will output predictions; \
                requires traintime 'train-val-split.json' to be in the same dir as the model",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="test the model; \
                will output evaluation metrics and predictions; \
                requires 'test_files.json' to be in the same dir as the model",
    )
    parser.add_argument(
        "-gc",
        "--gradcam",
        action="store_true",
        help="will output GradCAM visualizations",
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
    # Parse commandline arguments
    args = parser.parse_args()
    model_path = args.model
    model_path_parent = str(Path(model_path).parent)
    dataset_dir = args.data_dir
    make_gradcam = args.gradcam
    inference_mode = args.inference
    test_mode = args.test
    thresh = args.threshold

    # Set detection configs
    print("\nSetting detection configs...")
    print("Loading model...")
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_list(args.opts)
    cfg.DATASEED = ""
    cfg.PARENTDIR = ""
    cfg.CATEGORICAL_MAP = []
    cfg.DETECTION_MODE = ""
    cfg.merge_from_file(os.path.join(model_path_parent, "config.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cat_map = {'neoplastic': 0} ## TODO: unhardcode this
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
        print("Note: The loaded model was trained for single ROI detection")
    else:
        print("Note: The loaded model was trained for multi ROI detection")
    #max_dim = cfg.INPUT.MAX_SIZE_TRAIN
    #eval_logger.info(
    #        f"Input images will be resized to {max_dim}px on the longest edge"
    #    )

    if not inference_mode:  # Validation mode
        print("Performing model validation/ testing...")
        output_dir = os.path.join(
            model_path_parent, f"evaluations_{datetime.now().strftime('%Y%m%d')}"
        )
        # Update img and anno paths if the paths to dataset has changed
        # (e.g., running train and eval on different computers)
        # Read back the validation data split
        if test_mode:
            data_ledger = "test_files.json"
        else:
            data_ledger = "../../dataseed/OSCC_TCGA_FullMix_1008.json"
        with open(os.path.join(model_path_parent, data_ledger), "r") as f:
            data = json.load(f)
            val_data = data["val"]
        sample_im_path = Path(val_data.get("images")[0])
        source_dataset_dir = str(sample_im_path.parent.parent)
        if Path(dataset_dir) != Path(source_dataset_dir):
            print(
                "Train time dataset dir doesn't match the current dataset dir.\
                Did you switch omputers?"
            )
            print("Attempting to update paths to validation data...", end="")
            for d in val_data:
                updated_paths = [
                    i.replace(source_dataset_dir, dataset_dir) for i in val_data[d]
                ]
                val_data[d] = updated_paths
            print("done!")
        print(val_data)
        # Register dataset and metadata
        reg_name = 'val'
        dataset_name = 'ROI_' + reg_name
        print(f"Registering dataset '{dataset_name}'...")
        TrainHelper().register_dataset(
            reg_name, val_data, cat_map
        )
        ROI_metadata = MetadataCatalog.get(dataset_name)
        cfg.DATASETS.TEST = dataset_name
        dataloader = build_detection_test_loader(
            cfg, cfg.DATASETS.TEST, mapper=eval_utils.EvalHelper.validation_mapper
        )
    else:  # Inference mode
        print("Performing inference on a new dataset...")
        output_dir = os.path.join(
            dataset_dir, f"predictions_{Path(dataset_dir).stem}_{datetime.now().strftime('%Y%m%d')}"
        )
        reg_name = "ROI_pred"
        print(f"Registering dataset '{reg_name}'...")
        inf_helper = eval_utils.InferenceHelper(cfg)
        DatasetCatalog.register(
            reg_name, lambda d=reg_name: inf_helper.get_inference_dset(dataset_dir)
        )
        ROI_metadata = MetadataCatalog.get(reg_name).set(
            thing_classes=sorted(cat_map, key=cat_map.get)
        )
        cfg.DATASETS.TEST = reg_name
        print("Building dataloader for inference...")
        dataloader = build_detection_test_loader(
            cfg, cfg.DATASETS.TEST, mapper=inf_helper.inference_mapper
        )
        
    # Make output dir
    if os.path.exists(output_dir):
        overwrite_ans = str(
            input(
                "The output folder already exists! Would you like to overwrite it (y/n)?")
        )
        if "n" in overwrite_ans.lower():
            next_ver = 2
            while os.path.exists(output_dir+f"_v{next_ver}"):
                next_ver += 1
            output_dir = output_dir+f"_v{next_ver}"
        else:
            print("The existing output folder will be removed")
            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)
    (f"All eval outputs will be saved to '{output_dir}'\n")

    # Setup logging
    eval_logger = setup_logger(
        name="ROI_eval",
        output=os.path.join(
            output_dir,
            f"{datetime.now().strftime('%Y%m%d')}.log"
            )
        )

    # Initialize the model
    model = DefaultPredictor(cfg).model
    eval_dataset = cfg.DATASETS.TEST
    thing_classes = ROI_metadata.thing_classes

    #  Begin evaluation
    if not inference_mode:
        # Prepare dataset for COCO API
        print("Begin evaluation using the COCO API...")
        coco_evaluator = COCOEvaluator(
            dataset_name=eval_dataset,
            # tasks = ('bbox'),
            distributed=False,
            output_dir=output_dir,
            max_dets_per_image=10,
        )
        coco_evaluator.reset()
        eval_utils.convert_to_coco_json(eval_dataset, output_dir)

    print("Creating visualizations and saving predictions to json ...")
    if make_gradcam:
        conv_layers = gradcam.get_conv_layers(model)
        print(f"This model has {len(conv_layers)} convlutional layers in the backbone.", end=" ")
        print("Which layer to build GradCAM from?")
        layer_selection = input("Enter the indices (default = -1, the last layer)") or "-1"
        layer_indices = list(map(int, layer_selection.strip(',').split()))
        target_layers = [conv_layers[i] for i in layer_indices]
        eval_logger.info(f"GradCAM will be generated from {target_layers}")
    for inputs in dataloader:
        test_img_id = inputs[0]['image_id']
        eval_logger.info(f"Processing {test_img_id}...")
        input_img = np.transpose(inputs[0]["image"].data.numpy(), [1, 2, 0])
        cam_images = {} # Initialize to hold cam images if requested; makes fig plotting easier
        model.eval()
        with torch.no_grad():
            preds = model(inputs)
            instances = preds[0]["instances"].to('cpu')
            scores = instances.scores
            pred_classes = instances.pred_classes
            unique_classes = np.unique(pred_classes)
        if torch.numel(scores) == 0:
            eval_logger.warning("No predictions for this image!")
            continue
        else:
            # Output instance counts per class
            pred_stats = eval_utils.describe_predictions(pred_classes, cat_map)
            eval_logger.info(";".join(f"    {k}: {v}" for k, v in pred_stats.items()))
        # Process GradCAM if requested
        if make_gradcam:
            for i in unique_classes:
                class_cam = gradcam.GenerateCam(model, inputs, i, target_layers)
                cam_images[i] = class_cam()
        
        # Create visualizations
        fig = plt.figure()
        fig.suptitle(f"Visualizations for {test_img_id}")
        fig_name = f"pred_{test_img_id}.png"
        num_plots = len(cam_images) + 1
        if cam_images:
            for i, (class_label, class_cam) in enumerate(cam_images.items()):
                ax_gradcam = fig.add_subplot(1, num_plots, i+1)
                for k, v in cat_map.items():
                    if v == class_label:
                        class_name = k
                        if '_' in class_name:
                            subfig_title = class_name.split('_')[1]
                        else:
                            subfig_title = class_name
                ax_gradcam.imshow(class_cam)
                ax_gradcam.set_title(f"GradCAM: {subfig_title}")
                ax_gradcam.set_axis_off()
        visualizer = Visualizer(
            input_img, metadata=ROI_metadata, scale=1.0, instance_mode=2
        )
        pred_vis = visualizer.draw_instance_predictions(instances)
        vis_image = pred_vis.get_image()
        ax_pred = fig.add_subplot(1, num_plots, num_plots)
        ax_pred.imshow(vis_image)
        ax_pred.set_title("Predicted ROIs")
        plt.savefig(os.path.join(output_dir, fig_name), format="png", dpi=300)
        plt.close()
        # Accumulate COCO evaluation metrics (only if in validation mode)
        if not inference_mode:
            coco_evaluator.process(inputs, preds)
        # Otherwise in inference mode, parse predictions
        # to QuPath-compatible json
        else:
            # 1. Invert crop transform to original size
            # NOTE The original pixels are not restored, only padded with blanks
            tx1, ty1, tx2, ty2 = inputs[0]['tissue_xyxy']
            th = ty2 - ty1
            tw = tx2 - tx1
            cropper = T.CropTransform(
                tx1, ty1, tw, th,
                orig_w=inputs[0]['src_im_width'],
                orig_h=inputs[0]['src_im_height']    
                )
            uncropper = cropper.inverse()
            inverted_img = uncropper.apply_image(input_img)
            # 2. Invert predicted boxes and rewrite instances with inverted results
            pred_boxes = instances.pred_boxes
            inverted_boxes = [uncropper.apply_box(box)[0] for box in pred_boxes]
            instances.pred_boxes = inverted_boxes
            # 3. Build and write json
            pred_json = eval_utils.build_json(inputs, instances, thing_classes)
            json_name = f"pred_{test_img_id}.json"
            with open(os.path.join(output_dir, json_name), "w") as json_file:
                json_file.write(pred_json)       
    if not inference_mode:
        coco_results = coco_evaluator.evaluate()
    print("\nAll done!\n")
