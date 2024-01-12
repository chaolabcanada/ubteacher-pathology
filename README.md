# Unbiased Teacher v2: Semi-supervised Object Detection for Anchor-free and Anchor-based Detectors

<img src="teaser/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is Chao Lab's implementation of <br>
**Unbiased Teacher v2: Semi-supervised Object Detection for Anchor-free and Anchor-based Detectors**<br>
[Yen-Cheng Liu](https://ycliu93.github.io/), [Chih-Yao Ma](https://chihyaoma.github.io/), [Zsolt Kira](https://www.cc.gatech.edu/~zk15/)<br>
The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR), 2022 <br>

[[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Unbiased_Teacher_v2_Semi-Supervised_Object_Detection_for_Anchor-Free_and_Anchor-Based_CVPR_2022_paper.pdf)] [[Project](https://ycliu93.github.io/projects/unbiasedteacher2.html)]

<p align="center">
<img src="teaser/teaser_utv2.png" width="85%">
</p>

## What have we changed?
- A hybrid implementation of features from unbiased-teacher and unbiased-teacher-v2
- Allows user input at run-time for labeled vs. unlabeled datasets
- Allows empty annos. in labeled datasets for control purposes
- QuPath annotation integration (see below)
- Class merging logic to generalize disease-specific labels (see below)
- Support for Mask-RCNN training

## Work in progress
- Fix some images / annos not registering properly
- Create cfg defaults for parameters which can only be one value (e.g. num_classes)

## To Do
- Categorical mapping
- Augmentation rework
- Create a more descriptive logger to describe training flow
- Reduce box-in-a-box predictions

## Feature wishlist
- Transparency as to which data are being loaded at a given time
- Support for more formats / models
- Overfitting prevention
- Better visualizer during eval.

## Usage

### Required Config Parameters

- Note: any unspecified config fields will automatically inherit COCO defaults which may cause unintended consequences
- View output/config.yaml to see the FULL config of a given training run
The following .yaml block gives descriptions of our implementation-specific params., but NOT every necessary param. is included!

```yaml
# UBTeacherV2 Main Branch Config Params.
_BASE_: "../Base-RCNN-FPN.yaml" # Inherits detectron2 model config. Make sure path is correct.
MODEL:
  META_ARCHITECTURE: "TwoStagePseudoLabGeneralizedRCNN" # Altered from original UBTeacherV2 to support Mask-RCNN
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl" # Supports .pth and .pkl to load pre-training
MASK_ON: True # If true, mask_head is in roi_heads
PROPOSAL_GENERATOR:
  NAME: "PseudoLabRPN" # Unchanged fromm UBTeacherV2
ROI_HEADS:
  NAME: "StandardROIHeadsPseudoLab" # Now with Mask-RCNN
  LOSS: "FocalLoss_BoundaryVar" # Focal loss is key for "unbiased" classifications
  NUM_CLASSES: 4 # N + 1 classes, originally 80 for COCO training
SOLVER:
  IMG_PER_BATCH_UNLABEL: 10 # Keep this non-zero to avoid unintended behavior
DATALOADER:
  SUP_PERCENT: 100.0 # We want to use all labeled data available unlike COCO demo
  FILTER_EMPTY_ANNOTATIONS: # Lack of anno. doesn't necessarily mean unlabeled in our use-case.
DATASETS:
  CROSS_DATASET: True # Semi-supervision
  TRAIN_LABEL: ("train_labeled",) # train_net2.py registers with these names, feel free to change
  TRAIN_UNLABEL: ("train_unlabeled",) # see above
  TEST: ("val",) # see above
SEMISUPNET:
  Trainer: "ubteacher_rcnn" # only RCNN is supported for semi-supervised currently

# New Params. - These ALL require specification!
NUMPY: True # Determines whether .npy or .jpg/.png is expected to control image loader
UNLABELED_DIR: /mnt/d/unlabeled_images # Only required when cross_dataset is true, path to unlabeled images
IMG_DIR: /mnt/d/labeled_images # Path to labeled images
ANNO_DIR: /mnt/d/QuPath_annos # Path to anno. directories - Currently anno subdirs must match names with their corresponding img subdirs.
FMT: .svs, .tif, .tiff # WSI reader to get dimensions of originals to rescale annos during training
TRAIN_FRACTION: 0.8 # We disliked the train/test split logic so we redid it
SET_SEED: True # Determines whether a seed is output
DATASEED: /mnt/d/dataseed/model.json # Specifies the path for the dataseed to be written
REGISTER: True # Choose your own dataset or use a COCO default in TRAIN_LABEL/UNLABEL/TEST
DEBUG: False # Floods a folder with sample training images at train time - Use with caution!!!
BOX_ONLY: False # Only load box annotations and not polygons
CLASS_CONVERTER: /mnt/d/class_conversions/neoplastic.json # Path to a json with a dict. (see below)
```

### Using the Class Converter

- Sometimes classes are overly informative for training
- We want to condense the classes for training to avoid 400 different classes with 3 images each
- Dict. will be flipped and values mapper back to keys. Annos. will be registered with keys as the class.

```yaml
{"key1" : ["value1", "value2", "value3", "value4"], "key2": ["value5", "value6"]}
```

### Numpy vs. Jpg/Png

- Our implementation uses numpy inputs but we retain support for the COCO standard image formats
- cfg.NUMPY controls the following:
```python
# In ubteacher/data/dataset_mapper.py and in BaselineTrainer's DatasetMapperTwoCropSeparateV1 (not currently used)

if self.isnumpy:
  image = np.load(dataset_dict["file_name"])

if self.isnumpy:
  if "sem_seg_file_name" in dataset_dict:
  sem_seg_gt = np.load(dataset_dict.pop("sem_seg_file_name"))

# Otherwise, default detectron2 read_image is used
```
  
  
  
This project is licensed under [MIT License](LICENSE), as found in the LICENSE file.
