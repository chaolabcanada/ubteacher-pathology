"""
Utility functions to work with ROI_train.py

@Version: 0.7.1
@Author: Jesse Chao, PhD
@Contact: jesse.chao@sri.utoronto.ca
"""

import os
import json
import copy
import random
import logging
from typing import Dict, Tuple, List, Set, Iterator, Union

#import cv2
import torch
import numpy as np
import tifffile as tf
import argparse
import matplotlib.pyplot as plt
from matplotlib import patches
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer

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

def vis_image_with_annos(image, annotations, output):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for anno in annotations:
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
    fig.savefig(output)
    plt.close()

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
        vis_image_with_annos(auginput.image, annos_transformed, os.path.join(out_dir, vis_file))
        
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
    

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, mapper=TrainUtil(cfg.INPUT.MAX_SIZE_TRAIN).custom_mapper
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=TrainUtil(cfg.INPUT.MAX_SIZE_TRAIN).test_mapper,
        )
        
        
### Stuff from ROI_train.py that should be in utils


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
