"""
Use a UBTeacher faster RCNN model to perform inference, where tissues are detected as lymph/non-lymph in whole-slide images.

TODO 
- test the script on set 3

@Version: 0.5
@Author: Chao Lab, SRI
@Contact: jesse.chao@sri.utoronto.ca
"""
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
from detectron2.data.samplers import InferenceSampler
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultPredictor, DefaultTrainer
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

    def get_dataset(self, img_dir: str) -> list:
        dataset = []
        for file in os.scandir(img_dir):
            if not file.name.startswith('.') and file.name.endswith(self.compatible_formats):
                entry = {}
                entry['file_name'] = file.path
                entry['image_id'] = file.name
                dataset.append(entry)
        return dataset
    
    def dataset_mapper(self, dataset_dict: dict) -> dict:
        entry = copy.deepcopy(dataset_dict)
        try:
            with tf.TiffFile(entry['file_name']) as wsi:
                train_helper = train_utils.TrainUtil(self.max_dim)
                top = train_helper.find_top(wsi)
                top_dim = top.shape
                base = wsi.series[0].levels[0]
                base_dim = train_utils.channel_last(base.shape)
            entry['image'] = torch.from_numpy(top.transpose(2, 0, 1).copy())
            entry['height'] = top_dim[0]
            entry['width'] = top_dim[1]
            entry['base_height'] = base_dim[0]
            entry['base_width'] = base_dim[1]
            entry['src_im_height'] = top_dim[0]
            entry['src_im_width'] = top_dim[1]
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
        dataset = MapDataset(dataset, self.dataset_mapper)
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

class PostProcess:
    def __init__(self, model_input, registered_metadata, pred_instances, iou_thresh: float=0.10) -> None:
        """
        Args
            model_input (dict): a single input to the model; 
                            must be on CPU;
                            details at: https://detectron2.readthedocs.io/en/stable/tutorials/models.html
            registered_metadata (dict): detectron2 dataset metadata (Metadata)
            pred_instances (detectron2.structures.Instances object): predicted instances
            iou_thresh (float):  threshold (0-1) for IOU to trigger merging
        Attributes
            
        """
        self.input = model_input
        self.metadata = registered_metadata
        self.pred_classes = [cls.item() for cls in pred_instances.pred_classes]
        self.boxes = [box.detach().numpy().tolist() for box in pred_instances.pred_boxes]
        self.iou_thresh = iou_thresh

    def sort_bboxes(self, boxes: list, method="left-right") -> list:
        """
        Sort a list of bounding boxes in order of 'method'
        
        Args
            boxes (list): [box1, box2, ...] whereas each box is represetned as [x1, y1, x2, y2]
            method (str): one of 'left-right', 'right-left', 'top-bottom', 'bottom-top'
        Returns
            list of sorted bounding boxes
        """
        if len(boxes) < 2:
            return boxes
        else:
            # Initialize the reverse flag and sort index
            reverse = False
            i = 0
            # If sorting in reverse
            if method == "right-left" or method == "bottom-top":
                reverse = True
            # If sorting against the y-coordinate rather than
            # the x-coordinate of the bounding box
            if method == "top-bottom" or method == "bottom-top":
                i = 1
            # Sort boxes. Default: left-to-right
            sorted_bboxes = sorted(boxes, key=lambda b:b[i], reverse=reverse)
            # return the list of sorted bounding boxes
            return sorted_bboxes

    def merge_bboxes(self, boxes: list) -> list:
        """
        Merge overlapping bounding boxes

        Args
            boxes (list): [box1, box2, ...] whereas each box is represetned as [x1, y1, x2, y2]
        Returns
            a list of merged boxes without overlap
        """
        box_arr1 = copy.copy(self.sort_bboxes(boxes))
        if len(box_arr1) == 0:
            return box_arr1
        box_arr2 = [box_arr1.pop(0)]
        while len(box_arr1) > 0:
            merge = False
            # Assign left box as A, right as B
            keys = ['x1', 'y1', 'x2', 'y2']
            remove_indices = []
            for n in range(len(box_arr1)):
                ref_box = box_arr2[-1]  # Use the last box in arr2 as reference
                boxes2compare = [ref_box, box_arr1[n]]
                A, B = [{k: v for k, v in zip(keys, box)} for box in boxes2compare]
                # If no overlap b/w these two boxes, move on
                if B['x1'] > A['x2'] or \
                    max(A['y1'], B['y1']) > min(A['y2'], B['y2']):
                    continue
                # Otherwise, calculate IOU
                else:
                    u_w = max(A['x2'], B['x2']) - min(A['x1'], B['x1'])
                    u_h = max(A['y2'], B['y2']) - min(A['y1'], B['y1'])
                    union = u_w * u_h
                    i_w = min(A['x2'], B['x2']) - B['x1']
                    i_h = min(A['y2'], B['y2']) - max(A['y1'], B['y1'])
                    intersect = i_w * i_h
                    iou = round(intersect / union, 2)
                    if iou >= self.iou_thresh:
                        # Merge boxes
                        merge = True
                        remove_indices.append(n)
                        new_x1 = A["x1"]
                        new_y1 = min(A['y1'], B['y1'])
                        new_x2 = new_x1 + u_w
                        new_y2 = new_y1 + u_h
                        merged_box = [new_x1, new_y1, new_x2, new_y2]
                        box_arr2.pop(-1)
                        box_arr2.append(merged_box)
            # After looping through box_arr1, remove boxes that have been merged from box_arr1
            box_arr1 = [i for n, i in enumerate(box_arr1) if n not in remove_indices]
            if not merge:  # If no merging event has occured
                # Add the 1st box from box_arr1 to box_arr2 for the next round
                box_arr2.append(box_arr1.pop(0))
        return box_arr2

    def process_predictions(self) -> dict:
        """
        Merge overlapping boxes for each class
        
        Returns
            a dictionary containing a list of 'labels' and a list of 'boxes'
        """
        # Group predicted bboxes by class
        class_to_boxes = {}
        for inst_class, inst_box in zip(self.pred_classes, self.boxes):
            class_to_boxes.setdefault(inst_class, []).append(inst_box)
        pred_labels = []
        pred_boxes = []
        for cls, boxes in class_to_boxes.items():
            merged_boxes = self.merge_bboxes(boxes)
            pred_labels.extend([self.metadata.thing_classes[cls]] * len(merged_boxes))
            pred_boxes.extend(merged_boxes)
        return {'labels': pred_labels, 'boxes': pred_boxes}

    def summarize_predictions(self, processed_predictions):
        """
        Summarize predictions for an image and log the results.

        This function checks if there are any predicted classes for an image. If there are none, it logs a warning.
        Otherwise, it counts the occurrences of each label in the predictions and logs the counts.

        Args:
            processed_predictions (dict): cleaned prediction results with keys 'labels' and 'boxes' 

        Returns:
            A dictionary mapping labels to their counts, or None if there are no predicted classes.
        """
        class_counts = {}
        for label in self.metadata.thing_classes:
            count = len([i for i in processed_predictions['labels'] if i == label])
            class_counts[label] = count
        return f"Predictions: {class_counts}"

    def build_json(self, processed_predictions: dict):
        """
        Convert predictions to QuPath-compatible json
        
        Args
            processed_predictions (dict): cleaned prediction results with keys 'labels' and 'boxes'

        Returns 
            JSON formmated str
        """
        # Upsample boxes to base WSI dimensions
        scaled_boxes = train_utils.scale_bboxes(
            processed_predictions['boxes'],
            (self.input['src_im_height'], self.input['src_im_width']),
            (self.input['base_height'], self.input['base_width'])
                        )
        qp_boxes = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]] 
                                for x1, y1, x2, y2 in scaled_boxes]
        labels = processed_predictions['labels']
        cmap = matplotlib.colormaps['jet']
        qp_annotations = []
        for n, (label, box) in enumerate(zip(labels, qp_boxes)):
            color = cmap(n*25)
            color_int = [int(x*255) for x in color[:3]]
            anno_dict = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [box]},
                "properties": {
                    "object_type": "annotation",
                    "name": label,
                    "color": color_int,
                    "classification": {"name": "TissueFinder", "colorRGB": -16776961},  # Blue
                    "isLocked": False,
                },
            }
            qp_annotations.append(anno_dict)
            
        return json.dumps(qp_annotations, indent=4)        
    
    def plot_preds(self, vis_items: dict, output_dir: str) -> None:
        """
        Plots multiple images in subplots and saves the combined plot.

        Args:
            image_id (str): identifier for the plot (used in the title)
            vis_items (dict): list of dicts where keys are 'title', 'image', and values are str and image of shape (H, W, 3) (RGB) in uint8 type
            output_dir (str): directory where the plot image will be saved
        """
        image_id = vis_items['image_id']
        subplot_titles = vis_items['subplot_titles']
        images = vis_items['images']
        assert len(subplot_titles) == len(images)

        fig = plt.figure()
        fig.suptitle(image_id)
        num_subplots = len(subplot_titles)
        for n, (title, image) in enumerate(zip(subplot_titles, images)):
            ax = fig.add_subplot(1, num_subplots, n+1)
            ax.imshow(image)
            ax.set_title(title)
        plt.rc('font', size=6) # Controls default text sizes
        plt.rc('xtick', labelsize=4)    # Fontsize of the tick labels
        plt.rc('ytick', labelsize=4)    # Fontsize of the tick labels
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{image_id}_pred.png"), format="png", dpi=300)
        plt.close()        
        
def process_item(output_dir, registered_metadata, input_output_tuple: tuple, **kwargs) -> None:
    """
    Process a single item from a batch of inputs and outputs.

    This function takes a tuple of an input item and its corresponding output item, processes the output item's instances, 
    summarizes the predictions, crops tissue regions and saves them as numpy arrays, creates visualizations, and writes 
    predicted tissue regions to a JSON file.

    Args
        output_dir (str): the output directory
        registered_metadata
        input_output_tuple (tuple): a single input and its corresponding model output (instances)
    Returns
        message (str):
    """
    cam_data = kwargs.get('cam_data', {})

    # Postprocess predictions
    input_item, instances = input_output_tuple
    img_name = os.path.splitext(input_item['image_id'])[0]
    message = f"Processed {img_name}. "
    post_processor = PostProcess(input_item, 
                                registered_metadata, 
                                instances,
                                iou_thresh=0.1)
    
    if (len(post_processor.pred_classes)) == 0:
        message += "No predictions for this image!"
        return message
    
    pred_data = post_processor.process_predictions()
        #combine scores into labels
    pred_data['labels'] = [f"{label} : {np.round(score, 3)}" for label, score in zip(pred_data['labels'], pred_data['scores'])]
    
    # Create visualizations
    visualizer = Visualizer(
                    np.transpose(input_item['image'].data.numpy(), [1, 2, 0]), 
                    metadata=registered_metadata,
                    scale=1.0)
    pred_vis = visualizer.overlay_instances(
                boxes=pred_data['boxes'],
                labels=pred_data['labels']
                )
    plot_keys = ['image_id', 'subplot_titles', 'images']
    plot_values = [img_name, ['detected_tissues'], [pred_vis.get_image()]]
    if cam_data:
        plot_values[1].extend(f"GradCAM: {metadata.thing_classes[cls]}" for cls in cam_data)
        plot_values[2].extend(cam_data.values())
    
    post_processor.plot_preds(dict(zip(plot_keys, plot_values)), output_dir)
    
    # Create QuPath json (predictions as annotations)
    with open(os.path.join(output_dir, f"pred_{img_name}.json"), "w") as json_file:
        json_file.write(post_processor.build_json(pred_data))
    
    # Summarize processed prediction results
    message += post_processor.summarize_predictions(pred_data)
    return message
    
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
        default=0.7,
        help="set detection threshold; higher=more stringent; default=0.7",
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

    # ---------------------------------------
    # Prepare dataset
    # ---------------------------------------
    data_helper = DatasetHelper(cfg)
    print(f"Registering dataset '{reg_name}'...")
    DatasetCatalog.register(
        reg_name, lambda d=reg_name: data_helper.get_dataset(dataset_dir)
    )
    data_count = len(DatasetCatalog.get(reg_name))
    print(f"    found {data_count} compatible WSIs")
    metadata = MetadataCatalog.get(reg_name).set(
        thing_classes=sorted(cat_map, key=cat_map.get)
    )
    # ---------------------------------------
    # Perform inference
    # ---------------------------------------
    # Log and time the script
    start_time = time.time()
    tf_logger = setup_logger(
        name="tissue_finder",
        output=os.path.join(
            output_dir,
            f"{time.strftime('%Y%m%d', time.localtime())}.log"
            )
        )
    tf_logger.info(f"Begin inference using num_workers={num_workers} and batch size={batch_size}")
    
    # Build dataloader for batch inference
    dataloader = data_helper.build_batch_dataloader(
        dataset=DatasetCatalog.get(reg_name)
    )
    try:
        if cfg.SEMISUPNET.Trainer == "ubteacher_rcnn":
            # Initialize the UBTeacher model -- use the teacher
            student_model = UBRCNNTeacherTrainer.build_model(cfg)
            teacher_model = UBRCNNTeacherTrainer.build_model(cfg)
            ens_model = EnsembleTSModel(teacher_model, student_model)
            checkpointer = DetectionCheckpointer(ens_model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            model = ens_model.modelStudent ## is this right?
            model.eval()
        else:
            model = DefaultTrainer.build_model(cfg)
            checkpointer = DetectionCheckpointer(model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            model.eval()
    except AttributeError:
        model = DefaultTrainer.build_model(cfg)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        model.eval()

    # Configure GradCAM if requested
    if make_gradcam:
        # how does this differ from above method...
        conv_layers = gradcam.get_conv_layers(model)
        target_layer = [conv_layers[-1]]
        tf_logger.info(f"GradCAM will be generated from {target_layer}")
        
    # Forward pass
    for inputs in dataloader:
        # Process GradCAM if requested, no multiprocessing to avoid CUDA out of memory
        if make_gradcam:
            for input_item in inputs:
                class_cams, output_item = gradcam.GenerateCam(model, input_item, target_layer)()
                pred_info = process_item(
                                output_dir, 
                                metadata,
                                (input_item, output_item),
                                cam_data = class_cams
                            )
                pred_info = [pred_info]
                """
                fig = plt.figure()
                fig.suptitle(input_item['image_id'])
                num_subplots = len(class_cams.items())
                for n, (cls, cam) in enumerate(class_cams.items()):
                    ax = fig.add_subplot(1, num_subplots, n+1)
                    ax.imshow(cam)
                    ax.set_title(f"GradCam: {metadata.thing_classes[cls]}")
                plt.savefig(os.path.join(output_dir, f"{input_item['image_id']}_gradcam.png"), format='png', dpi=150)
                plt.close()
                """
                torch.cuda.empty_cache()
        else:        
            with torch.no_grad():
                batch_preds = model(inputs)
            outputs = [i['instances'].to('cpu') for i in batch_preds]
            # Postprocess predictions
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
            with mp.Pool(processes=num_workers) as pool:
                func = partial(
                    process_item,
                    output_dir,
                    metadata,
                )
                pred_info = pool.map(func, zip(inputs, outputs))
        for i in pred_info:
            tf_logger.info(i)
        
    tf_logger.info(f"All done! Total run time to process {data_count} images is {round(time.time() - start_time, 2)} seconds")