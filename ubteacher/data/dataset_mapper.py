# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import os #HACK
import matplotlib.pyplot as plt #HACK
from matplotlib import patches #HACK
import sys

import detectron2.data.detection_utils as utils
from detectron2.utils.file_io import PathManager
import detectron2.data.transforms as T
import numpy as np
import logging
import json
import torch
import time
from detectron2.data.dataset_mapper import DatasetMapper
from PIL import Image
from ubteacher.data.detection_utils import build_strong_augmentation
from ubteacher.utils.utils import vis_image_with_annos #TODO: Use this to visualize the dataset for debugging

sys.path.append(os.path.join(os.getcwd(), "pathology_he_auto_augment", "he_randaugment"))

from pathology_he_auto_augment.he_randaugment.augmenters.color.hedcoloraugmenter import HedColorAugmenter
import random


def build_augmentation(cfg, is_train):
        """
        Use our version instead of detectron2's version so we can mess with it
        Create a list of default :class:`Augmentation` from config.
        Now it includes resizing and flipping.

        Returns:
            list[Augmentation]
        """
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)] #TODO: Replace this with resize LONGEST edge
        if is_train and cfg.INPUT.RANDOM_FLIP != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )
            
        # NEW STUFF
        new_transforms = [
            #T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            #T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomLighting(0.5),
            T.RandomBrightness(0.8, 1.6), 
            T.RandomContrast(0.8, 1.6),
            ]
        augmentation.extend(new_transforms)
        return augmentation    
    
def hed_augmentation(image, factor):
    image=np.transpose(image,[2,0,1])
    augmentor= HedColorAugmenter(haematoxylin_sigma_range=(-factor, factor), haematoxylin_bias_range=(-factor, factor),
                                            eosin_sigma_range=(-factor, factor), eosin_bias_range=(-factor, factor),
                                            dab_sigma_range=(-factor, factor), dab_bias_range=(-factor, factor),
                                            cutoff_range=(0.15, 0.85))
    #To select a random magnitude value between -factor:factor, if commented the m value will be constant
    augmentor.randomize()
    return np.transpose(augmentor.transform(image),[1,2,0])

    
class DatasetMapperTwoCropSeparate(DatasetMapper):
    
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
        self.augmentation = build_augmentation(cfg, is_train)
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
        self.debug = cfg.DEBUG
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
        dataset_dict = copy.deepcopy(dataset_dict)  # dataset_dict is a slice of a list now
        if self.isnumpy:
            image = np.load(dataset_dict["file_name"]) #np.load instead
        else:
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)

        if "sem_seg_file_name" in dataset_dict:
            if self.isnumpy:
                sem_seg_gt = np.load(dataset_dict.pop("sem_seg_file_name"))
            else:
                sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation) #Modify aug_input
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

        
        # apply strong augmentation
        # We use torchvision augmentation, which is not compatiable with
        # detectron2, which use numpy format for images. Thus, we need to
        # convert to PIL format first.
        value = random.randint(0, 9)
        if value > 6: # 1/3 of the time apply d2 strong aug
            image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
            #image_strong_aug = hed_color_augmenter(image_strong_aug, (0, 0), (0, 0))
            image_strong_aug = np.array(self.strong_augmentation(image_pil))
        elif value > 3: # 1/3 of the time apply hsv / hed randaug
            image_strong_aug = hed_augmentation(image_weak_aug, 0.05)
        else: # Remainder are untransformed
            image_strong_aug = image_weak_aug  
            
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        )
         
               ## visualize the dataset for debugging
        if self.debug:
            out_dir = os.path.join(os.getcwd(), 'training_data_vis')
            os.makedirs(out_dir, exist_ok=True)
            n = 0
            vis_file = str(dataset_dict['image_id']) + '.png'
            while os.path.exists(os.path.join(out_dir, vis_file)):
                n += 1
                vis_file = f"{dataset_dict['image_id']}_{n}.png"
                if n > 2: #For only 2 augmentations
                    break
            try:    
                vis_image_with_annos(image_strong_aug, annos, os.path.join(out_dir, vis_file))
            except:
                pass

        dataset_dict_key = copy.deepcopy(dataset_dict)
        dataset_dict_key["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )
        assert dataset_dict["image"].size(1) == dataset_dict_key["image"].size(1)
        assert dataset_dict["image"].size(2) == dataset_dict_key["image"].size(2)
        return (dataset_dict, dataset_dict_key)

class TestMapper(DatasetMapper):
    def __init__(self, cfg):
        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.isnumpy = cfg.NUMPY
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.isnumpy = cfg.NUMPY
        
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # dataset_dict is a slice of a list now - unknown
        # USER: Write your own image loading if it's not from a file - ok I did
        if self.isnumpy:
            image = np.load(dataset_dict["file_name"]) #np.load instead
        else:
            image = utils.read_image(dataset_dict["file_name"])
            utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if self.isnumpy:
            if "sem_seg_file_name" in dataset_dict:
                sem_seg_gt = np.load(dataset_dict.pop("sem_seg_file_name"))
            else:
                sem_seg_gt = None
        else:
            if "sem_seg_file_name" in dataset_dict:
                sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
            else:
                sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals. - I don't

        return dataset_dict