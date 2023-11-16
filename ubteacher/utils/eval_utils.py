"""
Utility functions to perform evaluation or inference using a trained ROI detection
model.

@Version: 0.2.1
@Author: Jesse Chao, PhD
@Contact: jesse.chao@sri.utoronto.ca
"""

import json
import os
import copy
from pyexpat import model
#import logging
from typing import Dict, Tuple, List, Set, Iterator

import torch
from torch import nn
import shapely
import numpy as np
import tifffile as tf
import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from detectron2.utils.logger import setup_logger
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data.datasets.coco import convert_to_coco_dict
from detectron2.evaluation import DatasetEvaluator
from detectron2.config import CfgNode

from . import tissue_finder as tissue_finder
# NOTE TissueFinder uses 'agg' backend for matplotlib!
from . import ROI_utils as train_utils


def describe_predictions(pred_classes: List, cat_map: Dict):
    avail_classes = sorted(cat_map.keys())
    counts_per_class = np.histogram(pred_classes, bins=len(avail_classes))[0]
    stats = dict(zip(avail_classes, counts_per_class))
    return stats


class EvalHelper:
        
    @staticmethod
    def validation_mapper(dataset_dict: List[Dict]):
        """Mapper function to help create a custom dataloader.
        For validation only.

        Args:
        dataset_dict -- detectron2's standard dataset dict
        """
        #out_dir = os.path.join(os.getcwd(), 'validation_data_vis')
        #os.makedirs(out_dir, exist_ok=True)
        
        dataset_dict = copy.deepcopy(dataset_dict)
        annotation = dataset_dict.pop('annotations')
        image = np.load(dataset_dict['file_name'])
        input_image = torch.from_numpy(image.transpose(2, 0, 1))
        dataset_dict['image'] = input_image
        dataset_dict['instances'] = utils.annotations_to_instances(
                annotation, image.shape[1:]
        )

        #vis_out = os.path.join(out_dir, dataset_dict['image_id'] + '.png')
        #train_utils.vis_image_with_annos(tissue_img, annos_for_tissue, vis_out)
        return dataset_dict

class InferenceHelper(EvalHelper):
    def __init__(self, cfg: CfgNode) -> None:
            self.max_dim = cfg.INPUT.MAX_SIZE_TRAIN  # The max num pixels of the longest edge
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

    def get_inference_dset(self, img_dir):
        dataset_dicts = []
        for file in os.scandir(img_dir):
            if not file.name.startswith(".") and file.name.endswith(self.compatible_formats):
                try:
                    with tf.TiffFile(file.path) as tiff:
                        train_helper = train_utils.TrainUtil(self.max_dim)
                        top = train_helper.find_top(tiff)
                        top_dim = top.shape
                        base = tiff.series[0].levels[0]
                        base_dim = train_utils.channel_last(base.shape)
                        tissues = tissue_finder.TissueFinder(top).get_tissue_roi()
                        for n, t in enumerate(tissues):
                            X, Y, W, H = t
                            entry = {}
                            entry["file_path"] = file.path
                            entry["file_name"] = file.name
                            entry["image_id"] = file.name + f"_{n}"
                            entry['tissue_xyxy'] = [X, Y, X+W, Y+H]
                            entry['ref_dim'] = top_dim
                            entry["base_height"] = base_dim[0]
                            entry["base_width"] = base_dim[1]
                            dataset_dicts.append(entry)
                except:
                    print(f"TiffFile processing error, skipping {file.path}")
                    continue
        return dataset_dicts

    def inference_mapper(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        tissue_box = dataset_dict['tissue_xyxy']
        # Process tissue image
        ref_dim = dataset_dict['ref_dim']
        # Calculate crop dimensions at each scale of the pyramid
        with tf.TiffFile(dataset_dict['file_path']) as tiff:
            if len(tiff.series[0].levels) > 1:
                # Use levels to load, most common
                pyramid_reader = tiff.series[0].levels
            else:
                # Use pages to load, rare
                pyramid_reader = tiff.pages
            for lvl_idx, level in enumerate(pyramid_reader):
                lvl_dim = train_utils.channel_last(level.shape)
                lvl_crop = train_utils.scale_bboxes([tissue_box], ref_dim, lvl_dim)[0]
                cx1, cy1, cx2, cy2 = [int(i) for i in lvl_crop]
                if (cx2-cx1)>self.max_dim or (cy2-cy1)>self.max_dim:
                    target = lvl_idx
                    continue
                else:
                    target = lvl_idx - 1
                    break
            # Read image
            try:
                im = pyramid_reader[target].asarray()
            except IndexError:
                im = pyramid_reader[lvl_idx].asarray()
        im = train_utils.channel_last(im)
        im_h, im_w = im.shape[:2]
        # Crop tissue image
        resized_tissue_box = train_utils.scale_bboxes([tissue_box], ref_dim, im.shape)[0]
        tx1, ty1, tx2, ty2 = [int(i) for i in resized_tissue_box]
        cropper = T.CropTransform(tx1, ty1, (tx2-tx1), (ty2-ty1)) # NOTE CropTransform(x0, y0, w, h) from target level
        tissue_img = cropper.apply_image(im)
        # Resize crop if it's still too big
        if tissue_img.size:
            if tissue_img.shape[0] > self.max_dim or tissue_img.shape[1] > self.max_dim:
                t_h, t_w = tissue_img.shape[:2]
                scale_factor = self.max_dim/max(t_h, t_w)
                new_th = int(t_h * scale_factor)
                new_tw = int(t_w * scale_factor)
                tissue_img = T.ResizeTransform(t_h, t_w, new_th, new_tw).apply_image(tissue_img)
                
                im_h = int(im_h * scale_factor)
                im_w = int(im_w * scale_factor)
                resized_tissue_box = train_utils.scale_bboxes([resized_tissue_box], im.shape, (im_h, im_w))[0]
        tissue_img_tensor = torch.from_numpy(tissue_img.transpose(2, 0, 1).copy())
        dataset_dict['image'] = tissue_img_tensor
        dataset_dict['height'] = tissue_img.shape[0]
        dataset_dict['width'] = tissue_img.shape[1]
        dataset_dict['tissue_xyxy'] = resized_tissue_box
        dataset_dict['src_im_height'] = im_h
        dataset_dict['src_im_width'] = im_w
        return dataset_dict
    
def BoxMerger(box_coords: dict, method="left-right", iou_thresh: int=75) -> dict:
    '''A function to merge overlapping bounding boxes''' 
    
    if type(box_coords) == dict:
        try:
            box_coords = (box_coords["ROI_neoplastic"]) #retain only the neoplastic boxes
            print(box_coords)
        except:
            return None
    else:
        raise TypeError("Input must be a dictionary of coordinates.")

    """Sort a list of bounding boxes in order of 'method'
    Args:
    bboxes -- a list of bboxes in format [X1Y1X2Y2]
    method -- one of 'left-right', 'right-left', 'top-bottom', 'bottom-top'
    Returns:
    a list of sorted boxes
    """
    if len(box_coords) < 2:
        return box_coords
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
        # Sort bounding boxes. Default: left-to-right
        sorted_bboxes = sorted(box_coords, key=lambda b:b[i], reverse=reverse)
    
    """Merge overlapping bounding boxes.
    Args:
    boxes -- list of boxes with format [x1, y1, x2, y2]
    iou_thresh -- threshold for IOU to trigger merging
    Return:
    a list of merged boxes without overlap
    """
    box_arr1 = sorted_bboxes
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
            boxes2compare = []
            for box in (ref_box, box_arr1[n]):
                x, y, x2, y2 = box
                boxes2compare.append(int(i) for i in [x, y, x2, y2])
            A = dict(zip(keys, boxes2compare[0]))
            B = dict(zip(keys, boxes2compare[1]))
            # Check if there's no overlap b/w these two boxes
            if B['x1'] > A['x2'] or \
                max(A['y1'], B['y1']) > min(A['y2'], B['y2']):
                continue
            else:
                # Calculate IOU
                u_w = max(A['x2'], B['x2']) - min(A['x1'], B['x1'])
                u_h = max(A['y2'], B['y2']) - min(A['y1'], B['y1'])
                union = u_w * u_h
                i_w = min(A['x2'], B['x2']) - B['x1']
                i_h = min(A['y2'], B['y2']) - max(A['y1'], B['y1'])
                intersect = i_w * i_h
                iou = (intersect / union) * 100
                if iou >= iou_thresh:
                    # Merge boxes
                    merge = True
                    remove_indices.append(n)
                    merged_box_xyxy = [A['x1'],
                                        min(A['y1'], B['y1']), 
                                        A['x2'], 
                                        max(A['y2'], B['y2'])]
                    box_arr2.pop(-1)
                    box_arr2.append(merged_box_xyxy)
                    break
                    
        # After looping through box_arr1, remove boxes that have been merged
        # or exhaustively compared from box_arr1
        box_arr1 = [i for n, i in enumerate(box_arr1) if n not in remove_indices]
        if not merge:  # If no merging event has occured
            # Add the 1st box from box_arr1 to box_arr2 for the next round
            box_arr2.append(box_arr1.pop(0))
            
    box_arr1 = {"ROI_neoplastic": box_arr1}
    box_arr2 = {"ROI_neoplastic": box_arr2}
        
    return box_arr2


def convert_to_coco_json(dataset_name: str, output_dir: str, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
    dataset_name -- reference from the config file to the catalogs
                    must be registered in DatasetCatalog and in detectron2's 
                    standard format
    output_file -- path of json file that will be saved to
    allow_cached -- if json file is already present then skip conversion
    """
    logger = setup_logger(name=__name__)
    coco_files = []
    for file in os.scandir(output_dir):
        if "coco" in file.name:
            coco_files.append(file.path)
    if coco_files:
        logger.info("Removing previously cached coco files...")
        for i in coco_files:
            logger.info(f"Removed {i}")
            os.remove(i)
    coco_dict = convert_to_coco_dict(dataset_name)
    coco_json = json.dumps(coco_dict, indent=4)
    coco_json_file = dataset_name + "_coco_format.json"
    with open(os.path.join(output_dir, coco_json_file), "w") as json_file:
        json_file.write(coco_json)


class Counter(DatasetEvaluator):
    """
    A simple evaluator to count how many instances are detected
    in the validation dataset
    """

    def reset(self):
        self.count = 0

    def process(self, inputs: list, outputs: list):
        for input, output in zip(inputs, outputs):
            num_instances = len(output["instances"])
            print(f"Detected {num_instances} instances in {input['image_id']}")
            self.count += num_instances

    def evaluate(self):
        return {"count": self.count}


def get_all_inputs_outputs(model, data_loader):
    """
    Helper function to load data and feed to the model for inference
    """
    for inputs in data_loader:  # If no batching, len(inputs)=1
        # Uncomment below if doing batch inference
        # sample = batch[0]['image']
        # batch_dict = [{'image':sample[i]} for i in range(sample.size()[0])]
        # yield batch_dict, model(batch_dict)
        model.zero_grad()
        with torch.no_grad():
            predictions = model(inputs)
        yield inputs, predictions


def build_json(inputs: list, instances: list, thing_classes: list, cls_name: str='prediction') -> str:
    """
    Convert predictions to QuPath-compatible json
    Args:
    inputs -- input into a detectron2 model
    pred_instances -- predicted instances of a detectron2 model
    thing_classes -- list of classes registered in MetadataCatalog
    Return: 
    JSON formmated str
    """
    # Upsample bboxes to original dimensions, extrapolate XYXY to 4 points of [X, Y]
    #bboxes = [box.numpy() for box in instances.pred_boxes]
    bboxes = instances.pred_boxes
    scaled_bboxes = train_utils.scale_bboxes(
        bboxes,
        (inputs[0]["src_im_height"], inputs[0]["src_im_width"]),
        (inputs[0]["base_height"], inputs[0]["base_width"])
        )
    qp_bboxes = []
    for box in scaled_bboxes:
        x1, y1, x2, y2 = box
        qp_bboxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
    # Get the rest of the data
    scores = instances.scores.numpy()
    classes = [thing_classes[scr] for scr in instances.pred_classes]
    cmap = get_cmap("jet")
    features = []
    # parse every polygon annotation into a features:list
    for i in range(len(qp_bboxes)):
        rgba = cmap(i * 25)
        rgb_int = [int(j * 255) for j in rgba[:3]]
        anno_dict = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [qp_bboxes[i]]},
            "properties": {
                "object_type": "annotation",
                "name": f"{classes[i]} {scores[i]:.2f}",
                "color": rgb_int,
                "classification": {"name": cls_name, "colorRGB": -16776961},  # Blue
                "isLocked": False,
            },
        }
        features.append(anno_dict)

    return json.dumps(features, indent=4)


class GradCAM(object):
    """ Class for extracting activations (i.e., features) and
    registering gradients from the targeted layer
    General steps:
    1: place a hook at the last conv layer to calculate the gradient
    2: backpropagate the classifications score through the model
    """

    def __init__(self, model: object) -> None:
        """
        Args:
        model: a detectron2.modeling object
        layer_name: the name of the last convolutional layer
        """
        self.model = model
        self.gradients = None  # Placeholder for the gradients
        self.activations = None  # Placeholder for the activations
        self.handlers = []
        self._register_hook()

    def get_last_conv_name(self) -> str:
        """ Get the last conv layer from the model """
        layer_name = None
        for layer_name, module in self.model.named_modules():
            #if "norm" in layer_name and "backbone" in layer_name:
            if isinstance(module, nn.Conv2d) and "backbone" in layer_name:
                target_layer = layer_name
        return target_layer

    def _get_activations_hook(self, module, input, output) -> None:
        """ Get activations (i.e., feature maps) of the last conv layer
        """
        self.activations = output.detach().cpu()

    def _get_grads_hook(self, module, input, output) -> None: 
        """ Get the gradients of target layer w.r.t prediction outputs 
            Not using backward hook to record gradients,
            see https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/activations_and_gradients.py
        """
        def _store_grad(grad):
            self.gradients = grad.detach().cpu()
        output.register_hook(_store_grad)

    # register the hook
    def _register_hook(self):
        for (layer_name, module) in self.model.named_modules():
            if layer_name == self.get_last_conv_name():
                print(layer_name)
                self.handlers.append(
                    module.register_forward_hook(self._get_activations_hook)
                )
                self.handlers.append(
                    module.register_forward_hook(self._get_grads_hook)
                )

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def get_cam(self) -> np.ndarray:
        gradients = self.gradients[0].data.numpy()
        activations = self.activations[0].data.numpy()
        # Compute the average gradients per channel
        weights = np.mean(gradients, axis=(1, 2))
        weighted_activations = activations * weights[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(weighted_activations, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU
        # Postprocess
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.remove_handlers()


class GradCAMPlusPlus(GradCAM):
    def __init__(self, model: object) -> None:
        super().__init__(model)

    def get_cam(self):
        gradients = self.gradients[0].data.numpy()
        activations = self.activations[0].data.numpy()
        grads_power_2 = gradients**2
        grads_power_3 = grads_power_2 * gradients
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 0.000001
        aij = grads_power_2 / (2*grads_power_2 + sum_activations[:, None, None]*grads_power_3 + eps)
        aij = np.where(gradients != 0, aij, 0)
        weights = np.maximum(gradients, 0) * aij
        sum_weights = np.sum(weights, axis=(1, 2))
        weighted_activations = activations * sum_weights[:, np.newaxis, np.newaxis]
        cam = np.sum(weighted_activations, axis=0)
        cam = np.maximum(cam, 0)
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

class GradCAM_multilayer:
    def __init__(self, model, target_layers):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        self.target_layers = target_layers
        self._register_hook()
    
    def _save_activation(self, module, input, output):
        self.activations.append(output.detach().cpu())

    def _save_gradient(self, module, input, output):
        def _store_grad(grad):
            self.gradients.append(grad.detach().cpu())
        self.handles.append(
            output.register_hook(_store_grad)
        )

    def _register_hook(self):
        for (layer_name, module) in self.model.named_modules():
            for i in self.target_layers:
                if layer_name.endswith(i):
                    self.handles.append(
                        module.register_forward_hook(self._save_activation)
                    )
                    self.handles.append(
                        module.register_forward_hook(self._save_gradient)
                    )
    
    def release(self):
        for handle in self.handles:
            handle.remove()

    def compute_cam_per_layer(self):
        activations_list = [a[0].data.numpy() for a in self.activations]
        grads_list = [g[0].data.numpy() for g in self.gradients]
        cam_per_target_layer = []
        for i in range(len(self.target_layers)):
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]
            layer_weights = np.mean(layer_grads, axis=(1, 2))
            #print(f"layer_weights: {layer_weights.shape}")
            #print(f"layer activations: {layer_activations.shape}")
            weighted_activations = layer_activations * layer_weights[:, np.newaxis, np.newaxis]
            layer_cam = np.sum(weighted_activations, axis=0)  # [H,W]
            #print(f"layer_cam b4 postprocessing: {layer_cam.shape}")
            # Postprocess
            layer_cam = np.maximum(layer_cam, 0)  # ReLU
            layer_cam -= np.min(layer_cam)
            layer_cam /= np.max(layer_cam)
            cam_per_target_layer.append(layer_cam)
        return cam_per_target_layer

    def get_cam(self):
        cams = self.compute_cam_per_layer()
        # Stack cams from all requested layers together
        cams = np.stack(cams, axis=0)
        #print(f"after concatenating: {cams.shape}")
        # Combine all cams and normalize
        final_cam = np.maximum(cams, 0)
        final_cam = np.mean(cams, axis=0)
        #print(f"final cam: {final_cam.shape}")
        final_cam -= np.min(final_cam)
        final_cam /= np.max(final_cam)
        return final_cam

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.release()


def resize_image(image, target_size):
    return cv2.resize(image, target_size)

def visualize_cam(raw_cam: np.ndarray, input_image) -> np.ndarray:
    """ Generate heatmap from cam and resize it to input size
    Args:
    raw_cam: [H, W]
    input_image: [H, W, C]
    """
    resized_cam = resize_image(raw_cam, (input_image.shape[1], input_image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam = 0.5*heatmap + 0.5*input_image
    cam -= np.max(np.min(cam), 0)
    cam /= np.max(cam)
    #cam = np.uint8(cam * 255)
    return cam


# Edited Visualization Script

def visualize_images(test_img_id, input_img: np.ndarray, instances, pred_coords, 
                     gt_coords, cat_map: dict, accuracy, output_dir) -> None:

    """
    Draw instance-level predictions on an image. Returns an image object.

    Args:
        input_img: The input image (WSI) to draw predictions on.
        instances: The instances returned by the model after inference.
        test_img_id: The image ID of the input image.
        cat_map: The category map of the model.
    """
    
    tac_map = {v: k for k, v in cat_map.items()} # reverse the category map to get the tissue type from the class index
    classes = instances.pred_classes.tolist()
    scores = instances.scores.tolist()
    overlap, enveloped = accuracy
    fig, ax_pred = plt.subplots()
    ax_pred.imshow(input_img)

    try:
        scores = scores.remove(np.where(classes == 0))
        classes = classes.remove(0)
    except:
        pass
    
    # label empty labels as non-lymph
    if not classes:
        ax_pred.set_title(f"Tissue Type: Non-Lymph", fontsize=7)
    # choose the correct label based on the highest class score
    #TODO: Remove neoplastic from the options -- maybe solved?
    else: 
        max_score_index = scores.index(max(scores))
        overall = classes[max_score_index]
        ax_pred.set_title(f"Tissue Type: {tac_map[overall].title()}", fontsize=7)
        
    plt.axis('off')
    fig.suptitle(f"Visualizations for {test_img_id}", fontsize=10)
    fig.text(0.45, 0.95, "Pred.", va='top', ha='left', color='red', fontsize=7)
    fig.text(0.55, 0.95, "GT", va ='top', ha='left', color='green', fontsize=7)
    if overlap == 'N/A':
        pass
    else:
        fig.text(0.5, 0.03, f"IoU: {np.round(float(overlap), 3)*100} % \
                \n Tumor Coverage: {np.round(float(enveloped), 3)*100} %", ha="center", fontsize=7)
    

    # check all labels; label gt
    try:
        if "ROI_neoplastic" in gt_coords:
            gt_coords = gt_coords["ROI_neoplastic"]
            if len(gt_coords[0]) > 1:
                for i in range(len(gt_coords)):
                    x1, y1, x2, y2 = gt_coords[i]
                    ax_pred.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='green', linewidth=2))
            else:
                pass
    except:
        print("No ground truth annotations.")
        
    #check for the existence of predictions; label pred
    try:
        if "ROI_neoplastic" in pred_coords:
            pred_coords = pred_coords["ROI_neoplastic"]
            if len(pred_coords[0]) > 1:
                for j in range(len(pred_coords)):
                    x1, y1, x2, y2 = pred_coords[j]
                    ax_pred.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2))
            else:
                pass
    except:
        print("No predictions.")
        
    fig.canvas.draw()
    fig_name = f"pred_{test_img_id}.png"

    plt.savefig(os.path.join(output_dir, fig_name), format="png", dpi=300)
    plt.close()

    return

def get_positions(instances: any, gt_path: str) -> list:
    """ 
    Returns a list of two dictionaries that contain the coordinates in x1, y1, x2, y2 format.
    
    The first dictionary contains the predicted coordinates for each class.
    The second dictionary contains the ground truth coordinates for each class.

    To be used during inference.

    Params:
        instances: Instances (from inference).
        gt_path: Path to json file containing ground truth annotations.
    """
    # tac_map = {v: k for k, v in cat_map.items()}
    classes = instances.pred_classes.tolist()
    boxes = instances.pred_boxes.tensor.tolist()

    # For text mode, process the coordinates in the text file
    pred_coords = {}
    pred_coords_list = []
    for i in range(len(classes)):
        if classes[i] == 0: # neoplastic
            pred_coords_list.append(boxes[i])
    pred_coords["ROI_neoplastic"] = pred_coords_list # for now, we're only running this on those confirmed as ROI_neoplastic

    # load gts from json file
    try:
        gt_json = json.load(open(gt_path)) # dictionary containing gt anno
    except:
        return pred_coords, None

    gt_coords = {} # Dictionary that lists bboxes per class in x1, y1, x2, y2 format

    # Get formatted coordinates for ground truths
    for instances in gt_json["box_dicts"]:
        # print(f"{instances}, {type(instances)}")
        # print(list(instances))

        for label, boxes in instances.items():
            if not label in gt_coords:
                gt_coords[label] = []
            gt_coords[label].append(boxes)

    return pred_coords, gt_coords

def calculate_iou(pred_coords: dict, gt_coords: dict) -> dict:
    """
    Calculates the unary union for each class. Receives a dictionary. Keys are the classes, values are a list of coordinates in x1, y1, x2, y2 format.
    """
    all_keys = ['ROI_neoplastic', 'non-lymph', 'lymph']
    # all_keys = list(set(list(pred_coords.keys()) + list(gt_coords.keys()))) # get all unique classes
    try:
        pred_coords = pred_coords["ROI_neoplastic"]
        gt_coords = gt_coords["ROI_neoplastic"]
        # create a shapely polygon that adds all the ground truth boxes together
        gt_poly = shapely.Polygon()
        for i in range(len(gt_coords)):
            x1, y1, x2, y2 = gt_coords[i]
            curr_gt_coords = ((x1, y1), (x1, y2), (x2, y2), (x2, y1))

            if len(gt_coords) == 1:
                gt_poly = shapely.Polygon(curr_gt_coords)
            else:
                gt_poly = gt_poly.union(shapely.Polygon(curr_gt_coords))
        
        # create a shapely polygon that adds all the predicted boxes together
        pred_poly = shapely.Polygon()
        for i in range(len(pred_coords)):
            x1, y1, x2, y2 = pred_coords[i]
            curr_pred_coords = ((x1, y1), (x1, y2), (x2, y2), (x2, y1))
            
            if len(pred_coords) == 1:
                pred_poly = shapely.Polygon(curr_pred_coords)
            else:
                pred_poly = pred_poly.union(shapely.Polygon(curr_pred_coords))
    
        # calculate the intersection over union between the two shapely polygons
        iou = gt_poly.intersection(pred_poly).area / gt_poly.union(pred_poly).area
        te = gt_poly.intersection(pred_poly).area / gt_poly.area
    except:
        try:
            gt_coords = gt_coords['ROI_neoplastic']
            iou = 'N/A'
            te = 'N/A'
        except:
            try:
                pred_coords = pred_coords['ROI_neoplastic']
                iou = 0
                te = 0
            except:
                iou = 'N/A'
                te = 'N/A'
      
    return iou, te