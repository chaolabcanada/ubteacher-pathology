import json
import os
import copy
from typing import Dict, Tuple, List, Set, Iterator, Union
import random
import logging
import time
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
from matplotlib import patches
import argparse
from PIL import Image

import torch #TODO: only import whats required if at all
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils  #TODO: remove this potentially
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data.datasets.coco import convert_to_coco_dict #is this useful?
from detectron2.data.dataset_mapper import DatasetMapper
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils.events import EventStorage
import detectron2.utils.comm as comm
from detectron2.engine import DefaultTrainer, hooks, SimpleTrainer, TrainerBase
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.train_loop import AMPTrainer
from fvcore.nn.precise_bn import get_bn_modules
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog


from ubteacher.data.detection_utils import build_strong_augmentation
from detectron2.evaluation import (
    COCOEvaluator
)
from ubteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
)

from detectron2.evaluation import (
    COCOEvaluator,
    verify_results,
)

from ubteacher.solver.build import build_lr_scheduler

### Section 1: Data Processing and Loading ###

def find_anno_dir(parent_dir: str) -> List[str]:
    """
    Find qupath exported annotations directory
    """
    
    if os.path.exists(os.path.join(parent_dir, 'xupath_annotations_latest')):
        return os.path.join(parent_dir, 'qupath_annotations_latest')
    else:
        anno_dirs = []
        for root, dirs, files in os.walk(parent_dir):
            for d in dirs:
                if 'annotations' in d:
                    anno_dirs.append(os.path.join(root, d))
        # user chooses if there are multiple annotation folders
        print('Found multiple annotation folders:')
        for i, anno_dir in enumerate(anno_dirs):
            print(f'{i}: {os.path.relpath(anno_dir, parent_dir)}')
        choice = input('Choose annotation folder index')
        if choice.isdigit() and int(choice) < len(anno_dirs):
            return anno_dirs[int(choice)]    
        else:
            raise ValueError('Annotation folder not found')

def channel_last(input: np.ndarray or tuple) -> np.ndarray or tuple:
    """Return the input in channel-last format
    Args:
    input -- np.ndarray if image or tuple of array.shape
    Return:
    image as ndarray but in channel-last (h, w, c)
    """
    if type(input) == np.ndarray:
        if input.shape[0] == 3:
            return input.transpose(1, 2, 0)
        else:
            return input
    if type(input) == tuple:
        if input[0] == 3:
            return tuple(input[i] for i in [1, 2, 0])
        else:
            return input
        
def resize_image(image, max_dim):
    """
    Args:
    image -- np.ndarray with shape (h, w, c)
    Return:
    resized_image -- np.ndarray with shape (h, w, c)
    """
    h, w = image.shape[:2]
    scale_factor = round((max_dim / max(h, w)), 2)
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    resized_image = T.ResizeTransform(h, w, new_h, new_w).apply_image(image)
    return resized_image


def vis_boxes(image: np.array, bboxes: List, box_names=None):   
    '''Return visuals for bboxes on image.
    Args:
    image - np.array with shape (h, w, c),
    bboxes - list of [x1, y1, x2, y2] coordinates,
    box_names - optional list of names for each box.
    '''
    fig, ax = plt.subplots()
    N = len(bboxes) * 5
    cmap = plt.cm.get_cmap('hsv', N)
    # Pad image with border
    padded = np.pad(image, pad_width=((5,5), (5,5), (0,0)), mode='constant', constant_values=0)
    ax.imshow(padded)
    # Draw bbox over all kept contours
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
                    (x1, y1), width, height, linewidth=1, edgecolor=cmap(i), facecolor='none'
                )
        ax.add_patch(rect)
        if box_names:
            ax.annotate(box_names[i], (x1, y1), color=cmap(i), ha="left", va="top")
    plt.show(); plt.close()
    
def scale_bboxes(bboxes: list, ref_dim, target_dim):
    """Scale bounding boxes in XYXY
    
    Args:
    bboxes -- a list of bboxes in the form [XYXY]
    ref_dim -- (h, w, c) or (h, w)
    target_dim -- ((h, w, c) or (h, w))
    Return:
    scaled_bboxes -- a list of scaled [XYXY]
    """
    x_scale = ref_dim[1] / target_dim[1]
    y_scale = ref_dim[0] / target_dim[0]
    scaled_bboxes = []
    for box in bboxes:
        x0 = int(box[0] / x_scale)
        y0 = int(box[1] / y_scale)
        x1 = int(box[2] / x_scale)
        y1 = int(box[3] / y_scale)
        scaled_bboxes.append([x0, y0, x1, y1])
    return scaled_bboxes
    
### Section 2: Data registration ### -- TODO: Redo all these classes to trim down

class AnnoUtil:
    def __init__(self, annotation_file: str):
        """A class to hold image annotations with
        convenience functions to parse various attributes
        such as bounding box coordinates, annotation types, etc..

        Args:
        annotation_file -- full path to the json annotation
        """
        with open(annotation_file) as jFile:
            self.image_annotations = json.load(jFile)

    def search_recursive(self, d: Dict, key: str) -> Iterator:
        """Helper function for finding which level of json annotations has
        the matching key.
        """
        for k, v in d.items():
            if isinstance(v, Dict):
                for match in self.search_recursive(v, key):
                    yield match
            if k == key:
                # generator function - saves in memory until called
                # (use for loop to call)
                yield v

    def get_box_name(self, annotation_instance: Dict) -> str:
        """Find and return the name of an annotated bounding box.

        Args:
        annotation_instance -- annotation for one bounding box
        Return:
        box_name -- the annotated name of the box
        """
        # Pass on polygons
        num_vertices = len(next(self.search_recursive(annotation_instance, "coordinates"))[0])
        if num_vertices > 5:
            #print("passing", num_vertices)
            return None
        else:
            for properties in self.search_recursive(annotation_instance, "properties"):
                if "name" in properties.keys():
                    property_name = properties["name"].upper()
                    if "ROI" in property_name:
                        cls_name = properties["classification"]["name"].lower()
                        box_name = f"{property_name}_{cls_name}"
                    else:
                        box_name = properties["name"].lower()
                    return box_name
                else:
                    return None

    def find_annotypes(self) -> Set:
        """Find and return a unique list of annotations
        for all bounding boxes in an image.
        """
        annotations = self.image_annotations['box_dicts'] # box_dicts are parsed by tissue_finder
        annotypes = set()
        for instance in annotations:
            for k, v in instance.items():
                if k:
                    annotypes.add(k)
        #annotypes = set([list(instance.keys())[0] for instance in annotations])
        return annotypes

    def find_bbox_coordinates(self, annotation_instance: Dict) -> Dict[str, List]:
        """Find and return the coordinates for an annotated bounding box.
        Args:
        annotation_instance -- annotation for one bounding box
        Return:
        bbox_dict -- A Dict of [ROI type, list of bbox coordinates in XYXY]
        """
        box_name = self.get_box_name(annotation_instance)
        # box_coord = annotation_instance["geometry"]["coordinates"]
        if box_name:
            box_coord = next(self.search_recursive(annotation_instance, "coordinates"))
            if len(box_coord) == 1: # Some polygons are broken into multiple sub-lists; this is a bandaid fix
                box_arr = np.array(box_coord[0])
                bbox = [
                    min(box_arr[:, 0]),
                    min(box_arr[:, 1]),
                    max(box_arr[:, 0]),
                    max(box_arr[:, 1]),
                ]
                bbox_dict = {box_name: bbox}
                return bbox_dict
            else:
                pass
        else:
            pass

    def parse_bboxes(self) -> List[Dict]:
        """Parse bbox coordinates into detectron2's expected format: BoxMode.XYXY_ABS;
        expect image_annotations in QuPath's exported format.
        return:
        bbox_list = a list of single-length dicts
                    each dict's key is the classification
                    each value is bbox coordinates in [XYXY]
        """
        bbox_list = []
        if not isinstance(self.image_annotations, list):
            image_annotations = [self.image_annotations]
        else:
            image_annotations = self.image_annotations
        for i in image_annotations:
            bbox_dict = self.find_bbox_coordinates(i)
            # Skip if no bbox_dict can be parsed
            if not bbox_dict:
                pass
            # Check if key for bbox_dict is not None
            elif list(bbox_dict.keys())[0]:
                bbox_list.append(bbox_dict)
        return bbox_list

    @staticmethod
    def scale_bbox_dicts(
        bbox_dicts: List[Dict], ref_dim: Tuple, target_dim: Tuple
    ) -> List[Dict]:
        """Scale bbox coordinates according to provided dimensions.

        Args:
        bbox_dicts -- a list of dicts; each dict is in name: [box coordinates]
                        in XYXY
        ref_dim -- (h, w, c) or (h, w)
        target_dim -- ((h, w, c) or (h, w))
        """
        box_names = []
        boxes = []
        for b in bbox_dicts:
            for name, box in b.items():
                box_names.append(name)
                boxes.append(box)
        scaled_bboxes = scale_bboxes(boxes, ref_dim, target_dim)
        scaled_boxdicts = [{name: box} for name, box in zip(box_names, scaled_bboxes)]
        return scaled_boxdicts

    @staticmethod
    def scale_annotations(annotations, ref_dim, target_dim):
        scaled_annos = annotations.copy()
        bboxes = [a['bbox'] for a in annotations]
        scaled_bboxes = scale_bboxes(bboxes, ref_dim, target_dim)
        for a, b in zip(scaled_annos, scaled_bboxes):
            a['bbox'] = b
        return scaled_annos

    def parse_annotations(self, bboxes: List[Dict], cat_map):
        annotations = []
        for i in bboxes:
            instance = {}
            for k, v in i.items():
                cat_name = k
                cat_id = cat_map[cat_name]
                instance["category_id"] = cat_id
                instance["bbox"] = v
                instance["bbox_mode"] = 0  # detectron2.structures.BoxMode(XYXY_ABS=0)
            annotations.append(instance)
        return annotations 

class TrainUtil:
    def __init__(self, max_dimension=1333) -> None:
        self.max_dim = max_dimension  # The max num pixels of the longest edge

    @staticmethod
    def train_val_split(
        image_dir: str,
        annotation_dir: str,
        split_miu: float = 0.2,
    ):
        """Split image dataset and the associated annotations into "train" and "test".
        WARNING: images and annotations must have the same name

        Args:
        image_dir -- path to images
        annotation_dir -- path to annotations (e.g., .json)
        compatible_formats -- ex: 'tif', 'jpg'
        split_miu -- fraction of overall data to split as validation dataset

        Return:
        train_set -- Dict('images': [paths], 'annotations': [paths])
        val_set -- Dict('images': [paths], 'annotations': [paths])
        """
        train = []
        train_unlabeled = []
        val = []
        anno_train = []
        anno_val = []

        for img_file in os.scandir(image_dir):
            # Find image
            if not img_file.name.startswith(".") and img_file.name.endswith('npy'):
                image_name = os.path.splitext(img_file.name)[0]
                # Find matching annotation json
                anno_file = os.path.join(annotation_dir, image_name + ".json")
                if os.path.exists(anno_file):  # Only add images with matching annos
                    anno_train.append(anno_file)
                    train.append(img_file.path)

        # Randomly split some validation data
        val_data_len = int(len(train) * split_miu)
        train_data_len = len(train) - val_data_len
        split_guide = [True] * val_data_len + [False] * train_data_len
        random.shuffle(split_guide)
        for idx, to_split in enumerate(split_guide):
            if to_split:
                split_guide.pop(idx)
                val.append(train.pop(idx))
                anno_val.append(anno_train.pop(idx))
        train_set = dict()
        val_set = dict()
        for i, data_group in zip(
            (train_set, val_set), ((train, anno_train), (val, anno_val))
        ):
            for n, k in enumerate(["images", "annotations"]):
                i[k] = data_group[n]
        return train_set, val_set


    def crop_image_to_dim(self, wsi_path: str, ref_box: List, ref_dim: Union[Tuple, List]) -> np.ndarray:
        """ Get an image crop from the correct pyramid level
        such that the ideal target dimensions are achieved in the crop
        Args:
        wsi_path -- path in str to the wsi
        ref_box -- the reference box for the cropping operation NOTE [XYXY]
        ref_dim -- the dimension of the reference image from which the ref_box was calculated
        Return:
        image as numpy array
        """
        with tf.TiffFile(wsi_path) as tiff:
            if len(tiff.series[0].levels) > 1:
                # Use levels to read
                pyramid_reader = tiff.series[0].levels
            else:
                # Use pages to read
                pyramid_reader = tiff.pages
            # Go through the pyramid, get dimensions of tissue crop at each level,
            # compare with max_dim, stop when the ideal level is reached
            for lvl_idx, level in enumerate(pyramid_reader):
                lvl_dim = channel_last(level.shape)
                lvl_crop = scale_bboxes([ref_box], ref_dim, lvl_dim)[0]
                cx1, cy1, cx2, cy2 = [int(i) for i in lvl_crop]
                crop_h = cy2 - cy1
                crop_w = cx2 - cx1
                if crop_w>self.max_dim or crop_h>self.max_dim:  # Are pyramid levels alwasys read from base to top?
                    target = lvl_idx
                    continue
                else:
                    target = lvl_idx -1
            # Read image
            try:
                img = pyramid_reader[target].asarray()
            except IndexError:
                img = pyramid_reader[lvl_idx].asarray()
            img = channel_last(img)
        # Crop image NOTE CropTransform(x0, y0, w, h)
        crop_coord = scale_bboxes([ref_box], ref_dim, img.shape)[0]
        tx1, ty1, tx2, ty2 = [int(i) for i in crop_coord]
        cropper = T.CropTransform(tx1, ty1, (tx2-tx1), (ty2-ty1))
        cropped_img = cropper.apply_image(img)
        # Resize valid tissue img if necessary
        if cropped_img.shape[0]>self.max_dim or cropped_img.shape[1]>self.max_dim:
            cropped_img = resize_image(cropped_img, self.max_dim)
        return cropped_img

    def find_top(self, tiff: tf.TiffFile) -> np.ndarray:
        """Return top of image pyramid under the specified resolutions; if
        embedded top level exceeds specified dimensions, will downsample.

        Args:
        tiff -- a tfffile.TiffFile image object
        """
        # GOAL: Get the highest level of image pyramid that is slightly bigger than
        # our target dimension, and resize the longest edge to target
        # Check if image pyramid is organized into levels (most files) or pages (rare)
        max_dim = self.max_dim
        top = np.array([])

        # Find the top of the image pyramid
        if len(tiff.series[0].levels) > 1:
            # Use levels to load
            for level_idx, level in enumerate(
                tiff.series[0].levels
            ):
                # Make sure image shape is channel-last
                level_dim = channel_last(level.shape)
                if level_dim[1] > max_dim or level_dim[2] > max_dim:
                    target_level = level_idx
                    continue
                else:
                    target_level = level_idx - 1
                    break
            top = tiff.series[0].levels[target_level].asarray()
        else:
            # Use pages to load
            for page_idx, page in enumerate(tiff.pages):
                # Make sure image shape is channel-last
                page_dim = channel_last(page.shape)
                if page_dim[1] > max_dim or page_dim[2] > max_dim:
                    target_page = page_idx
                    continue
                else:
                    target_page = page_idx - 1
                    break
            top = tiff.pages[target_page].asarray()
        # Downsample if the top is still bigger than requested
        # Make sure image array is in shape[h, w, c]
        top = channel_last(top)
        if top.shape[1] > max_dim or top.shape[2] > max_dim:
            h, w = top.shape[:2]
            longest_edge = max(h, w)
            scale_factor = round((max_dim / longest_edge), 2)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            resized = T.ResizeTransform(h, w, new_h, new_w).apply_image(top)
            return resized
        else:
            return top

    def get_annotation_dicts(
        self, dataset: Dict, cat_map: Dict
    ) -> List[Dict]:
        """Custom metadata parser to read json annotations and
        prepare them into detectron2's expected format.

        Args:
        dataset -- Dict with keys 'images' and 'annotations', where
                   images: list of path (type:str) to images
                   annotations: list of path (type:str) to json annotations
        cat_map -- Dict to map each bbox annotation to their corresponding
                   categorical id
        """
        logger = logging.getLogger("ROI_train")
        images = dataset["images"]
        annotations = dataset["annotations"]
        dataset_dicts = []
        # Read image metadata (annotations, etc) from a json file
        for img_path, anno_path in zip(images, annotations):
            anno_helper = AnnoUtil(anno_path)
            record = {}
            record['file_name'] = img_path  # Full path to the corresponding image file
            record['height'] = anno_helper.image_annotations['image_height']
            record['width'] = anno_helper.image_annotations['image_width']
            record['image_id'] = os.path.splitext(os.path.basename(anno_path))[0]
            try:
                boxes = anno_helper.image_annotations['box_dicts']
            except KeyError:
                continue
            # Build the "annotations" field as required by detectron2
            try:
                annotations = anno_helper.parse_annotations(boxes, cat_map)
            except KeyError:
                logger.warning(
                            f"The annotation in {img_path} is not in categorical mapping!"
                        )
        else:
            annotations = []
                
            record["annotations"] = annotations
            # Add to dataset_dicts
            dataset_dicts.append(record)
        return dataset_dicts
    
    def basic_anno_dicts(
        self, dataset: Dict) -> List[Dict]:
        """For use when annotations are prepared in 
        the correct format already"""
        logger = logging.getLogger("ROI_train")
        images = dataset["images"]
        annotations = dataset["annotations"]
        dataset_dicts = []
        #Read image metadata (annotations, etc) from the new json format
        for img_path, anno_path in zip(images, annotations):
            anno_helper = AnnoUtil(anno_path)
            record = {}
            record['file_name'] = img_path  # Full path to the corresponding image file
            record['height'] = anno_helper.image_annotations['image_height']
            record['width'] = anno_helper.image_annotations['image_width']
            record['image_id'] = os.path.splitext(os.path.basename(anno_path))[0]
            record['annotations'] = anno_helper.image_annotations['annotations']
            dataset_dicts.append(record)
        return dataset_dicts

    def custom_mapper(self, dataset_dict: List[Dict]):
        """Mapper function to help create a custom dataloader.

        Args:
        dataset_dict -- detectron2's standard dataset dict
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        try:
            annotations = dataset_dict.pop('annotations')
        except:
            return dataset_dict
        image = np.load(dataset_dict['file_name'])
        augs = T.AugmentationList(
            [
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomBrightness(0.7, 1.3),
            T.RandomContrast(0.7, 1.3),
            T.RandomSaturation(0.7, 1.3),
            
            # WARNING! RandomRotation might cause loss to go to infinity!
            # RandomCrop doesn't work either
            # Disable if it causes issues.
            # T.RandomRotation([-10, 10], sample_style="choice", expand=False),
            ]
        )
        # (2) Apply transformations
        auginput = T.AugInput(image)
        transform = augs(auginput)  # type: T.Transform
        # (3) Return transformed images and their annotations
        # NOTE: model expects image in (c, h, w)
        image_transformed = torch.from_numpy(auginput.image.transpose(2, 0, 1))
        annos_transformed = [
            utils.transform_instance_annotations(
                anno, [transform], image_transformed.shape[1:]
            )
            for anno in annotations
        ]
        # Uncomment the following lines to visualize augmented training images
        out_dir = os.path.join(os.getcwd(), 'training_data_vis')
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        vis_file = dataset_dict['image_id'] + '.png'
        while os.path.exists(os.path.join(out_dir, vis_file)):
            n += 1
            vis_file = f"{dataset_dict['image_id']}_{n}.png"
        
        dataset_dict['image'] = image_transformed
        dataset_dict['instances'] = utils.annotations_to_instances(
                annos_transformed, image_transformed.shape[1:]
        )
        return dataset_dict
    
    def test_mapper(self, dataset_dict: List[Dict]):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = np.load(dataset_dict["file_name"])
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
        dataset_dict["image"] = image_tensor
        return dataset_dict

class TrainHelper(TrainUtil):
    # def __init__(self):
    #    super().__init__()
    def get_unlabeled(self, cfg, unlabeled_dataset) -> Dict:
        
        parent_dir = cfg.PARENTDIR
        image_dir = os.path.join(parent_dir, unlabeled_dataset)       
        annotation_dir = os.path.join(image_dir, "tissue_annotations")
        
        unlabeled_imgs = []
        unlabeled_annos = []
        for img_file in os.scandir(image_dir):
            if not img_file.name.startswith(".") and img_file.name.endswith('npy'):
                    image_name = os.path.splitext(img_file.name)[0]
                    # Find matching annotation json
                    anno_file = os.path.join(annotation_dir, image_name + ".json")
                    if os.path.exists(anno_file):  # Only add images with matching annos
                        unlabeled_annos.append(anno_file)
                        unlabeled_imgs.append(img_file.path)
                

                
        data_unlabeled = {'images': unlabeled_imgs, 'annotations': unlabeled_annos}     
        
        return data_unlabeled

    def split_dataset(self, cfg, dataset_names: List[str], args: argparse.Namespace, set_seed = False):
        """Function to collect all input datasets and split them
        into 'train' and 'val' sets.
        Args:
        dataset_names: a list of dataset names in the dataset directory
        args: parsed command-line arguments
        
        """
        data_train = dict()
        data_val = dict()
        parent_dir = cfg.PARENTDIR

        for name in dataset_names:
            image_dir = os.path.join(parent_dir, name)
            annotation_dir = os.path.join(image_dir, "tissue_annotations")

            #Try loading from dataseed
        try:
            with open(cfg.DATASEED, 'r') as f:
                data = json.load(f)
                train_set = data['train']
                val_set = data['val']
        except:
            
            print('Dataseed did not load. Creating new seed.')
                            
            train_set, val_set = self.train_val_split(
                image_dir, annotation_dir, 
            )         
        for i, j in zip((data_train, data_val), (train_set, val_set)):
            for k, v in j.items():
                if k in i:
                    i[k].extend(v)
                else:
                    i[k] = v
                        
            # Create json specifying train/val split
        if set_seed:
            data = {'train': train_set, 'val': val_set}
            with open(cfg.DATASEED, 'w') as f:
                json.dump(data, f)
                
        return data_train, data_val
    
    def preview_data(self, dataset, num_sample: int, out_dir: str) -> None:
            """Save preview data as .png to out_dir.

            Args:
            dataset -- either "data_train" or "data_val" as returned by split_dataset()
            num_samples -- number of training data to sample < len(dataset)
            out_dir -- output directory to save the previews to
            """
            num_images = len(dataset["images"])
            for n, i in enumerate(random.sample(range(0, num_images), num_sample)):
                img_file = dataset["images"][i]
                print(f"previewing {img_file}")
                curr_img = np.load(img_file)
                curr_anno = dataset["annotations"][i]
                anno_helper = AnnoUtil(curr_anno)
                bboxes = anno_helper.image_annotations['box_dicts']
                cmap = plt.cm.get_cmap('hsv', len(bboxes)*2)
                fig, ax = plt.subplots()
                ax.imshow(curr_img)
                for c, b in enumerate(bboxes):
                    for box_name, box_coord in b.items():
                        x1, y1, x2, y2 = [int(i) for i in box_coord]
                        w = x2 - x1
                        h = y2 - y1
                        rect = patches.Rectangle(
                            (x1, y1), w, h, linewidth=1, edgecolor=cmap(c), facecolor="none"
                        )
                        ax.add_patch(rect)
                        if c % 2 == 0:
                            Y = y1
                        else:
                            Y = y2
                        ax.annotate(
                            box_name, (x1, Y), color=cmap(c), fontsize=6, ha="left", va="bottom"
                        )
                fig.suptitle(img_file)
                ax.set_xlabel("pixels")
                ax.set_ylabel("pixels")
                fig_name = f"train-sample_{n}.png"
                plt.savefig(os.path.join(out_dir, fig_name), dpi=300, format="png")
                plt.close()

    def register_dataset(
        self, dset_type: str, dataset: Dict, cat_map: Dict
    ) -> None:
        """Helper function to register a new dataset to detectron2's
        Datasetcatalog and Metadatacatalog.

        Args:
        dataset -- dict with keys 'images' and 'annotations', where each value is a list of
        [image_paths] and [annotation_paths]}, in which
                image_paths -- list of paths to image files
                annotation_paths -- list of paths to annotation files
        cat_map -- dictionary to map categories to ids, e.g. {'ROI':0, 'JUNK':1}
        """
                
        images = dataset["images"]
        annotations = dataset["annotations"]
        reg_name = "ROI_" + dset_type
        
        # Register dataset to DatasetCatalog
        print(f"working on '{reg_name}'...")
        
        if len(images) != len(annotations):
            print(
                f"There are {len(images)} images but {len(annotations)} annotations, "
                "you may want to double check if this was expected..."
            )
        DatasetCatalog.register(
            reg_name,
            lambda d=dset_type: self.basic_anno_dicts(dataset), #Changed this to basic anno dicts
        )
        # Register metadata to MetadataCatalog
        MetadataCatalog.get(reg_name).set(
            thing_classes=sorted([k for k, v in cat_map.items()])
        )
        return MetadataCatalog
    
    # def basic_registration(self, dset_type: str, dataset: Dict):
     #   reg_name = "ROI_" + dset_type
     #   print(f"working on '{reg_name}'...")
     #   DatasetCatalog.register(reg_name, lambda d=dset_type: self.basic_anno_dicts(dataset))
     #   MetadataCatalog.get(reg_name).set(thing_classes=['lesion'])

def get_annotypes_for_dataset(dataset: Dict) -> Set:
    """Get all annotation types (names of annotations)
    in the given dataset
    """
    dset_annotypes = set()
    for anno in dataset["annotations"]: #TODO: Fix this
        anno_helper = AnnoUtil(anno)
        try:
            annotypes_per_im = anno_helper.find_annotypes()
        except KeyError:
            continue
        dset_annotypes.update(annotypes_per_im)
    return dset_annotypes    


def get_categorical_map(detection_mode: str, global_annotypes: Set) -> Dict:
    """Generate a dict to map annotated category to id"""
    cat_map = dict()
    cat_code = 0
    for i in sorted(list(global_annotypes)):
        if detection_mode == "multi":
            cat_map[i] = cat_code
            cat_code += 1
        if detection_mode == 'single':
            main_class = i.split("_")[0]
            if main_class not in cat_map:
                cat_map[main_class] = cat_code
                cat_code += 1
    return cat_map

### Part 3: Misc. debugging tools ### 

def vis_image_with_annos(image, annotations, output):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for anno in annotations:
        #plot bbox
        x1, y1, x2, y2 = anno['bbox']
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle(
            (x1, y1),
            w, h,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.annotate(f"class_ID={anno['category_id']}", (x1, y1), color='b')
        #plot poly
        x_coords = anno['segmentation'][0][::2]
        y_coords = anno['segmentation'][0][1::2]
        ax.plot(x_coords, y_coords, color='g')
    fig.savefig(output)
    plt.close()

def write_batch(batch, output_dir):
    batch.append(time.time())
    # write json
    json_file = os.path.join(output_dir, 'batch_log.json')
    with open(json_file, 'w') as f:
        json.dump(batch, f)  
        
        
### Part 4: Customizations from ubteacher v1 that need to be addressed ###

class DatasetMapperTwoCropSeparateV1(DatasetMapper):
    """
    This customized mapper produces two augmented images from a single image
    instance. This mapper makes sure that the two augmented images have the same
    cropping and thus the same size.

    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """


    
    def __init__(self, cfg, is_train=True):
        self.augmentation = utils.build_augmentation(cfg, is_train)
        # include crop into self.augmentation
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            self.compute_tight_boxes = True
        else:
            self.compute_tight_boxes = False
        self.strong_augmentation = build_strong_augmentation(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.isnumpy = cfg.NUMPY
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if self.isnumpy:
            image = np.load(dataset_dict["file_name"]) #numpy version
        else:
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            

        if "sem_seg_file_name" in dataset_dict:
            if self.isnumpy:
                sem_seg_gt = np.load(
                    dataset_dict.pop("sem_seg_file_name"), "L"
                ).squeeze(2)
            else:
                sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
                
        else:
            sem_seg_gt = None

        # apply weak augmentation
        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image_weak_aug.shape[:2]  # h, w

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

            if self.compute_tight_boxes and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            bboxes_d2_format = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = bboxes_d2_format
            
        # Plot weakly augmented inputs to prove correct transformation #HACK
        
        #out_dir = os.path.join(os.getcwd(), 'training_data_vis')
        #os.makedirs(out_dir, exist_ok=True)
        #n = 0
        #vis_file = dataset_dict['image_id'] + '.png'
        #while os.path.exists(os.path.join(out_dir, vis_file)):
        #    n += 1
        #    vis_file = f"{dataset_dict['image_id']}_{n}.png"
        #    if n > 2: #For only 2 augmentations
        #        break
        #vis_image_with_annos(image_weak_aug, annos, os.path.join(out_dir, vis_file))

        # apply strong augmentation
        # We use torchvision augmentation, which is not compatiable with
        # detectron2, which use numpy format for images. Thus, we need to
        # convert to PIL format first.
        
        image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
        image_strong_aug = np.array(self.strong_augmentation(image_pil))
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        )

        dataset_dict_key = copy.deepcopy(dataset_dict)
        dataset_dict_key["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )
        assert dataset_dict["image"].size(1) == dataset_dict_key["image"].size(1)
        assert dataset_dict["image"].size(2) == dataset_dict_key["image"].size(2)
        return (dataset_dict, dataset_dict_key)

class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        
        # create only one model 
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # Fr training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
            
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if cfg.TEST.EVALUATOR == "COCOeval":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        # elif cfg.TEST.EVALUATOR == "COCOTIDEeval":
        #     return COCOTIDEEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparateV1(cfg, True)
        #can be replaced with a custom mapper if needed
        return build_detection_semisup_train_loader(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)
    
    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start
        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)
        
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)