import glob
import json
import os
import argparse
from typing import Dict, Tuple, List, Set, Iterator
import multiprocessing as mp
import numpy as np
import tifffile as tf # type: ignore
import matplotlib # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib import patches # type: ignore
from detectron2.data import transforms as T # type: ignore
# from utils.train_utils import AnnoUtil, TrainUtil, channel_last, scale_bboxes, resize_image
# TODO: change the copy back to og once things work
import utils.train_utils_copy as train_utils

#extra stuff that i am importing
from openslide import OpenSlide # type: ignore
from PIL import Image # type: ignore
# To run: python annotation_preprocessing_copy.py /mnt/RSX/Datasets_pathology/SRI_OSCC_lymph_labeled /home/adrit/home/unbiased_teacher2/tem --qupath_annos /mnt/RSX/Datasets_pathology/SRI_OSCC_lymph_labeled/qupath_annotations_latest/

# Authors: Chao Lab, 2024

'''
This script is designed to create ground truths for training a ubteacher model. 
We are interested in both labeled and unlabeled data, since every image needs an annotation.
From the input images, we are cropping out each tissue and its corresponding annotations,
With 1 .npy and 1. json per tissue.

Unlabeled data has no annotations, but we still need to crop out the tissue and save it as a .npy file.
This is done based on the predictions from TissueFinder, which must be specified.

Class conversions condense the annotations into manageable classes: currently neoplastic, non-neoplastic, and ignore.
These are specified in a json file, which must be specified.


Improvements:
- Add support for other image formats # THIS CAN BE ADDED ONCE WE FULLY FIGURE OUT TIFFFILE
- Add support for tissue masking
- Automatically find the annotation folder (default: qupath_annotations_latest)
    - First check if qupath_annotations_latest exists, if not, prompt the user with a list of the folders in the parent directory
    - The user can then choose TF preds instead 
- Add support for cat map -> we want to load the cat_map and assign classes in "process_tissue_polygons"
- Support if class conversions are not provided
- Create better defaults
- Create better arg parse (i.e. help, required args replacing optional etc.)
- Refactor for readability, functionality is fine
- Macenko normalization

- Multi-process i/o function
'''

######################## MY OWN HELPER FUNCTIONS BELOW
def convert_openslide_to_dict(reader) -> dict:
    """
    Converts an OpenSlide reader object into a dictionary with a format similar to tiatoolbox's info output.

    Parameters:
    - reader: OpenSlide object

    Returns:
    - A dictionary containing slide information in the same format as tiatoolbox's read.info.as_dict()
    """
    # Extract information from OpenSlide properties
    properties = reader.properties
    slide_dimensions = (int(properties.get('openslide.level[0].width')),
                        int(properties.get('openslide.level[0].height')))

    level_count = int(properties.get('openslide.level-count'))
    level_dimensions = []
    level_downsamples = []

    for i in range(level_count):
        level_width = int(properties.get(f'openslide.level[{i}].width'))
        level_height = int(properties.get(f'openslide.level[{i}].height'))
        downsample = float(properties.get(f'openslide.level[{i}].downsample'))

        level_dimensions.append((level_width, level_height))
        level_downsamples.append(downsample)

    # Parse other relevant properties
    mpp_x = float(properties.get('openslide.mpp-x', '0.0'))
    mpp_y = float(properties.get('openslide.mpp-y', '0.0'))
    mpp = (mpp_x, mpp_y)

    objective_power = float(properties.get('openslide.objective-power', '-1'))

    vendor = properties.get('openslide.vendor', 'unknown')

    # Construct the output dictionary
    slide_info = {
        'objective_power': objective_power,
        'slide_dimensions': slide_dimensions,
        'level_count': level_count,
        'level_dimensions': tuple(level_dimensions),
        'level_downsamples': level_downsamples,
        'vendor': vendor,
        'mpp': mpp,
        'file_path': reader._filename,
        'axes': 'YXS'  # Assuming images are in Height x Width x Channels format
    }

    return slide_info


def dimensions_index(info_dict: dict) -> int:
    downsampling_factor = round(0.5/info_dict['mpp'][0])
    print("dwnsmpl" + str(downsampling_factor))
    for i in range(0, len(info_dict['level_downsamples'])):
        int(info_dict['level_downsamples'][i])
        if downsampling_factor == int(info_dict['level_downsamples'][i]):
            return i, downsampling_factor # index, value
    return -1, -1


def auto_detect_annotation_dir(src_dir):
    """
    If `qupath_annotations_latest` exists in `src_dir`, return it.
    Otherwise, prompt the user from a list of directories found in `src_dir`.
    """
    # Candidate default
    candidate = os.path.join(src_dir, 'qupath_annotations_latest')
    if os.path.exists(candidate) and os.path.isdir(candidate):
        print(f"Using default annotation folder: {candidate}")
        return candidate
    else:
        # Gather directories in src_dir (excluding the images themselves)
        # Typically, your images might be *.svs, *.tif, etc.
        # We'll list only subfolders:
        subdirs = [
            d for d in os.listdir(src_dir)
            if os.path.isdir(os.path.join(src_dir, d))
        ]
        if not subdirs:
            print(f"No subdirectories found in {src_dir}.")
            return None
        else:
            print(
                "No 'qupath_annotations_latest' folder found. "
                "Please choose from the following directories:"
            )
            for idx, d in enumerate(subdirs):
                print(f"  {idx + 1}. {d}")

            choice = input("Enter the number of the annotation directory to use: ")
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(subdirs):
                    chosen_dir = os.path.join(src_dir, subdirs[choice_idx])
                    print(f"Using {chosen_dir} as annotation directory.")
                    return chosen_dir
                else:
                    print("Invalid choice.")
                    return None
            except ValueError:
                print("Invalid input.")
                return None


def pick_level_that_fits(info_dict: dict, max_dim: int) -> int:
    """
    Given an info_dict describing a WSI's pyramid levels (from OpenSlide),
    return an integer index identifying which level does not exceed
    the given max_dim (on both width and height).

    Heuristic:
      1. Iterate from level 0 (full resolution) down to the last level
         (highest downsampling).
      2. Find the largest level that fits within max_dim.
      3. If none fit, return the index of the smallest level.

    Parameters
    ----------
    info_dict : dict
        A dictionary from `convert_openslide_to_dict(reader)` or similar,
        containing the keys:
          - 'level_dimensions': Tuple of ( (W0, H0), (W1, H1), ..., (Wn, Hn) )
          - 'level_count':      Number of levels
    max_dim : int
        The maximum allowed size (in pixels) for width/height.

    Returns
    -------
    int
        The best level index.
    """

    # e.g., info_dict['level_dimensions'] might be:
    #   ((50000, 30000), (25000, 15000), (12500, 7500), ...)
    levels = info_dict['level_dimensions']
    level_count = info_dict['level_count']

    chosen_level = level_count - 1  # default to the smallest level

    for level_idx in range(level_count):
        w, h = levels[level_idx]
        # If both sides fit in max_dim, this is a candidate.
        if w <= max_dim and h <= max_dim:
            chosen_level = level_idx
            break  # because we want the largest level that still fits

    return chosen_level


def pick_level_that_fits_tiff(pyramid_reader, base_dim: Tuple[int, int], max_dim: int) -> int:
    """
    Chooses which level from TiffFile's pyramid_reader best fits under max_dim.
    'pyramid_reader' is typically a list/tuple of pages or levels.

    Returns an integer index of the chosen level.
    """
    chosen_level = len(pyramid_reader) - 1  # default to the smallest level
    for idx, level_page in enumerate(pyramid_reader):
        # shape might be (height, width, channels) or (height, width)
        level_shape = level_page.shape
        # If the array has channels last, shape is (H, W, C), else (H, W)
        if len(level_shape) == 3:
            h, w, _ = level_shape
        else:
            h, w = level_shape

        if w <= max_dim and h <= max_dim:
            chosen_level = idx
            break  # pick the largest level that fits
    return chosen_level


def get_level_scale_factor(info_dict: dict, target_level: int) -> float:
    """
    Returns how much smaller the target_level is compared to level 0.
    If level 0 is 50000x30000 and level 2 is 12500x7500,
    then scale factor = 0.25.
    """
    base_w, base_h = info_dict['level_dimensions'][0]
    t_w, t_h = info_dict['level_dimensions'][target_level]
    # They might not exactly match the same ratio, but we’ll approximate
    scale_w = t_w / base_w
    scale_h = t_h / base_h
    # Typically these should be nearly identical, but let's pick one 
    # or average them if you want
    return (scale_w + scale_h) / 2.0

#################################### ENDS HERE


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

    
def get_bbox(poly):
    x = poly[0::2]
    y = poly[1::2]
    return np.min(x), np.min(y), np.max(x), np.max(y)


def get_polygon(poly):
    x = poly[0::2]
    y = poly[1::2]
    return np.array(list(zip(x, y)))


def process_unlabeled(image_file, annos, max_dim, out_dir, base_dim, info_dict, is_human_labeled):
    image_id = os.path.basename(image_file).split(".")[0]
    with tf.TiffFile(image_file) as slide:
        for n, a in enumerate(annos):
            # Calculate cropped tissue dimensions at each level of the image pyramid
            for k, v in a.items():
                tissue_coords_base = v
                tx1, ty1, tx2, ty2 = tissue_coords_base
            # print("    ", f"annotation is '{k}'")

            if not use_tiff:
                # print(info_dict)
                reader = OpenSlide(image_file)
                for level_idx in range(len(info_dict['level_dimensions'])):
                    level_dim = info_dict['level_dimensions'][level_idx]
                    level_dim = level_dim[::-1] # match the tifffile format
                    level_crop = train_utils.scale_bboxes([tissue_coords_base], base_dim, level_dim)[0]
                    cx1, cy1, cx2, cy2 = [int(i) for i in level_crop]
                    # print("level_dim is ^^^^^^^^^^^^^^^^", level_dim)
                    # print("level_crop is ^^^^^^^^^^^^^^^", level_crop)
                    crop_w = cx2-cx1
                    crop_h = cy2-cy1
                    if crop_h>max_dim or crop_w>max_dim:
                        target_level = level_idx
                        continue
                    else:
                        target_level = level_idx - 1
                        break
                something = info_dict['level_dimensions'][target_level]
                input_image = reader.read_region(location=(0, 0), level=target_level, size=something).convert('RGB')
                # print(input_image)
                # print("##################################")
                input_image = np.array(input_image)
            else:
                if len(slide.series[0].levels) > 1:
                    pyramid_reader = slide.series[0].levels
                else:
                    pyramid_reader = slide.pages
                for level_idx, level in enumerate(pyramid_reader):
                    level_dim = train_utils.channel_last(level.shape)
                    level_crop = train_utils.scale_bboxes([tissue_coords_base], base_dim, level_dim)[0]
                    cx1, cy1, cx2, cy2 = [int(i) for i in level_crop]
                    crop_w = cx2-cx1
                    crop_h = cy2-cy1
                    if crop_h>max_dim or crop_w>max_dim:
                        target_level = level_idx
                        continue
                    else:
                        target_level = level_idx - 1
                        break

                input_image = pyramid_reader[target_level].asarray()
                input_image = train_utils.channel_last(input_image)
            # Recrop image NOTE CropTransform needs input in XYWH
            recrop_coord = train_utils.scale_bboxes([tissue_coords_base], base_dim, input_image.shape)[0]
            rx1, ry1, rx2, ry2 = [int(i) for i in recrop_coord]
            recropper = T.CropTransform(rx1, ry1, rx2-rx1, ry2-ry1)
            recropped_tissue = recropper.apply_image(input_image)
            # Resize crop if necessary
            if recropped_tissue.shape[0]>max_dim or recropped_tissue.shape[1]>max_dim:
                #print("        ", 
                    #f"this tissue crop is bigger than {max_dim}px",
                    #"and will be downsampled")
                recropped_tissue = train_utils.resize_image(recropped_tissue, max_dim)
            #else:
                #print("        ", "done!")
            
            tissue_anno_dict = {
                    "file_name": f"{image_file.split('.')[0]}_{n}.npy",
                    "image_id": f"{image_id}_{n}",
                    "original_width" : base_dim[1],
                    "original_height" : base_dim[0],
                    "tissue_xyxy" : tissue_coords_base,
                    "width" : recropped_tissue.shape[1],
                    "height" : recropped_tissue.shape[0],
                    "max_dim": max_dim,
                    "annotations": [],
                    "labeled" : str(is_human_labeled)
                }
        
            if not os.path.exists(os.path.join(out_dir, 'tissue_annotations')):
                os.makedirs(os.path.join(out_dir, 'tissue_annotations'))
            anno_json = json.dumps(tissue_anno_dict, indent = 4, cls=NpEncoder)
            with open(os.path.join(out_dir, 'tissue_annotations', f"{image_id}_{n}.json"), 'w') as j:
                j.write(anno_json)
            
            if not os.path.exists(os.path.join(out_dir, 'visualizations')):
                    os.makedirs(os.path.join(out_dir, 'visualizations'))
                        
            # Visualize
            fig, ax = plt.subplots()
            ax.imshow(recropped_tissue)  
            plt.savefig(os.path.join(out_dir, 'visualizations', f'{image_id}_{n}.png'))
            plt.close()
            
             # For each tissue, save to disk with annotations in json
            print("    ", "Saving to disk...")
            np.save(os.path.join(out_dir, f"{image_id}_{n}.npy"), recropped_tissue)
    return
        
    
def process_tissue_polygons(class_conv, image_file, annos, lesions, max_dim, base_dim, out_dir, use_tiff, info_dict, is_human_labeled, cat_map=None, v3_annos=False):
    # TODO: BELOW THING IS UNUSED, INVESTIGATE WHERE AND HOW TO USE THIS
    # # If cat_map is provided, load it. Otherwise, fallback. 
    # if cat_map and os.path.exists(cat_map):
    #     with open(cat_map) as f:
    #         cat_mapping = json.load(f)  # e.g. {"non-neoplastic":0,"neoplastic":1,"ignore":2}
    # else:
    #     # Default fallback
    #     cat_mapping = {"non-neoplastic":0,"neoplastic":1,"ignore":2}

    image_id = os.path.basename(image_file).split(".")[0]
    with tf.TiffFile(image_file) as slide:
        for n, a in enumerate(annos):
            # Calculate cropped tissue dimensions at each level of the image pyramid
            for k, v in a.items():
                tissue_coords_base = v
                tx1, ty1, tx2, ty2 = tissue_coords_base
                base_long_edge = max(tx2-tx1, ty2-ty1)
            if not use_tiff:
                # print(info_dict)
                reader = OpenSlide(image_file)
                for level_idx in range(len(info_dict['level_dimensions'])):
                    level_dim = info_dict['level_dimensions'][level_idx]
                    level_dim = level_dim[::-1] # match the tifffile format
                    level_crop = train_utils.scale_bboxes([tissue_coords_base], base_dim, level_dim)[0]
                    cx1, cy1, cx2, cy2 = [int(i) for i in level_crop]
                    # print("level_dim is ^^^^^^^^^^^^^^^^", level_dim)
                    # print("level_crop is ^^^^^^^^^^^^^^^", level_crop)
                    crop_w = cx2-cx1
                    crop_h = cy2-cy1
                    if crop_h>max_dim or crop_w>max_dim:
                        target_level = level_idx
                        continue
                    else:
                        target_level = level_idx - 1
                        break
                something = info_dict['level_dimensions'][target_level]
                input_image = reader.read_region(location=(0, 0), level=target_level, size=something).convert('RGB')
                # print(input_image)
                # print("##################################")
                input_image = np.array(input_image)

            else:
                try:
                    pyramid_reader = slide.series[0].levels
                except:
                    try:
                        pyramid_reader = slide.pages
                    except:
                        print(f"Could not find pyramid reader for {image_id}")
                for level_idx, level in enumerate(pyramid_reader):
                    level_dim = train_utils.channel_last(level.shape)
                    level_crop = train_utils.scale_bboxes([tissue_coords_base], base_dim, level_dim)[0]
                    cx1, cy1, cx2, cy2 = [int(i) for i in level_crop]
                    # print("level_dim is", level_dim)
                    # print("level_crop is ", level_crop)
                    crop_w = cx2-cx1
                    crop_h = cy2-cy1
                    if crop_h>max_dim or crop_w>max_dim:
                        target_level = level_idx
                        continue
                    else:
                        target_level = level_idx - 1
                        break
                # print(pyramid_reader[target_level])
                input_image = pyramid_reader[target_level].asarray() # Could this be the clue to replace openslide???
                input_image = train_utils.channel_last(input_image)
                # region_image = Image.fromarray(input_image)
                # print(region_image)
            
            # Read image at the correct level
            # print("target is", target_level)
            # print(input_image)
            # Recrop image NOTE CropTransform needs input in XYWH
            recrop_coord = train_utils.scale_bboxes([tissue_coords_base], base_dim, input_image.shape)[0]
            rx1, ry1, rx2, ry2 = [int(i) for i in recrop_coord]
            recropper = T.CropTransform(rx1, ry1, rx2-rx1, ry2-ry1)
            recropped_tissue = recropper.apply_image(input_image)
            # Resize crop if necessary
            if recropped_tissue.shape[0]>max_dim or recropped_tissue.shape[1]>max_dim:
                #print("        ", 
                    #f"this tissue crop is bigger than {max_dim}px",
                    #"and will be downsampled")
                recropped_tissue = train_utils.resize_image(recropped_tissue, max_dim)
            #else:
                #print("        ", "done!")
            new_long_edge = max(recropped_tissue.shape[0], recropped_tissue.shape[1])
            scale_factor = round(new_long_edge/ base_long_edge, 3)
            
            if not os.path.exists(os.path.join(out_dir, 'visualizations')):
                os.makedirs(os.path.join(out_dir, 'visualizations'))
                        
            # Visualize
            print("visualizing")
            fig, ax = plt.subplots()
            ax.imshow(recropped_tissue)  
            plt.savefig(os.path.join(out_dir, 'visualizations', f'{image_id}_{n}.png'))
            plt.close()

             # For each tissue, save to disk with annotations in json
            print("saving visuals")
            np.save(os.path.join(out_dir, f"{image_id}_{n}.npy"), recropped_tissue)
            
            annotation_dicts = []
            
            # Scale and offset lesion annotations (polygons) to tissue
            processed_polygons = []
            polygon_names = []
            poly_bboxes = []
            for le in lesions:
                for k, v in le.items():
                    polygon = np.array(v)
                    polygon_names.append(k)
                # Scaling
                scaled_polygon = np.multiply(polygon, scale_factor)          
                # Offsetting
                tx1, ty1, tx2, ty2 = tissue_coords_base
                tissue_offset = np.array([[tx1, ty1]])
                offset_polygon = np.subtract(
                    scaled_polygon, 
                    np.tile(tissue_offset*scale_factor, (scaled_polygon.shape[0], 1))
                    )
                offset_x = np.max([i[0] for i in offset_polygon])
                offset_y = np.max([i[1] for i in offset_polygon])
                
                if np.any(offset_y>recropped_tissue.shape[0]) or \
                    np.any(offset_x>recropped_tissue.shape[1]) or \
                    np.any(offset_polygon<0): 
                    continue
                if not np.array_equal(offset_polygon[0], offset_polygon[-1]): # Drop open polygons
                    #print("        ", "dropped an open polygon")
                    continue
                else:
                    processed_polygons.append(offset_polygon) 
                    # Get bbox for each polygon
                    poly_bbox = get_bbox(offset_polygon)
                    poly_bboxes.append(poly_bbox)
                    
            if not os.path.exists(os.path.join(out_dir, 'visualizations')):
                os.makedirs(os.path.join(out_dir, 'visualizations'))
                        
            # Visualize
            fig, ax = plt.subplots()
            ax.imshow(recropped_tissue)  
            for poly in processed_polygons:
                ax.plot(poly[:, 0], poly[:, 1])
            plt.savefig(os.path.join(out_dir, 'visualizations', f'{image_id}_{n}.png'))
            plt.close()
            
            # Round to int and flatten
            processed_polygons = ([np.round(i).astype(int) for i in processed_polygons])
            processed_polygons = ([i.flatten() for i in processed_polygons])

             # For each tissue, save to disk with annotations in json
            print("    ", "Saving to disk...")
            np.save(os.path.join(out_dir, f"{image_id}_{n}.npy"), recropped_tissue)
            #Create annotation dict
            annotation_dicts = []
            for c, poly in enumerate(processed_polygons):
                #TODO: replace with cat_map 
                '''
                if polygon_names[c].lower() in class_conv['non-neoplastic']:
                    num = 0
                elif polygon_names[c].lower() in class_conv['neoplastic']:
                    num = 1
                elif polygon_names[c].lower() in class_conv['ignore']:
                    num = 2
                else:
                    continue   
                '''               
                poly_bbox = get_bbox(poly)
                if v3_annos:
                    each_anno = {
                        "label" : polygon_names[c],
                        "bounding_box": poly_bbox,
                    }
                else:
                    each_anno = {
                        "category_id" : 0, # TODO: Change this based on above 216-225
                        "bbox" : poly_bbox,
                        "bbox_mode" : 0,
                    # "segmentation" : [poly]
                    }
                annotation_dicts.append(each_anno) 
            
            # At the end of this: we want to save a "cat_map" -> json file which maps the class names to the class numbers
            # {0: "non-neoplastic", 1: "neoplastic", 2: "ignore"}  
            if v3_annos:
                tissue_anno_dict = {
                    "file_path": f"{image_id}.npy",
                    "image_id": image_id,
                    "original_width": base_dim[1],
                    "original_height": base_dim[0],
                    "width": recropped_tissue.shape[1],
                    "height": recropped_tissue.shape[0],
                    "annotations": annotation_dicts,
                    "labeled": str(is_human_labeled)
                }
                print(tissue_anno_dict)
                raise Exception
            tissue_anno_dict = {
                "file_name": f"{image_file.split('.')[0]}_{n}.npy",
                "image_id": f"{image_id}_{n}",
                "original_width" : base_dim[1],
                "original_height" : base_dim[0],
                "tissue_xyxy" : tissue_coords_base,
                "width" : recropped_tissue.shape[1],
                "height" : recropped_tissue.shape[0],
                "max_dim": max_dim,
                "annotations": annotation_dicts,
                "labeled" : str(is_human_labeled)
            }
            if not os.path.exists(os.path.join(out_dir, 'tissue_annotations')):
                os.makedirs(os.path.join(out_dir, 'tissue_annotations'))
            anno_json = json.dumps(tissue_anno_dict, indent = 4, cls=NpEncoder)
            with open(os.path.join(out_dir, 'tissue_annotations', f"{image_id}_{n}.json"), 'w') as j:
                j.write(anno_json)
    return


def lesion_finder_gt(src_dir, out_dir, image_path, max_dim, annos_dir, use_tiff, is_human_labeled, valid_tissues=None, label='neoplastic'):
    if valid_tissues is None:
        valid_tissues = []
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    image_id = image_path.split('/')[-1].split('.')[0]
    image_ext = '.' + image_path.split('/')[-1].split('.')[1]
    image_file = os.path.join(src_dir, image_id+image_ext)
    
    if not use_tiff:
        reader = OpenSlide(image_file)
        info_dict = convert_openslide_to_dict(reader)
        dimension_index, downsampling_factor = dimensions_index(info_dict)
        base_x, base_y = info_dict['level_dimensions'][dimension_index]
        base_dim = info_dict['slide_dimensions'][::-1 ]
        print("openslide base_dim", base_dim)

    else:
        # Read WSI header and get some info
        with tf.TiffFile(image_file) as slide:
            # Set image dimension attributes
            try:
                base_dim = slide.series[0].levels[0].shape
                base_dim = train_utils.channel_last(base_dim)
            except:
                # load image
                image = tf.imread(image_file)
                base_dim = train_utils.channel_last(image.shape)
        info_dict = {}

        print("tiff base_dim", base_dim)
    # Read annotations
    if os.path.exists(os.path.join(annos_dir, image_id+".json")):
        anno_helper = train_utils.AnnoUtil(os.path.join(annos_dir, image_id+".json"))
        image_annotations = anno_helper.image_annotations
        is_labeled = True
    elif os.path.exists(os.path.join(annos_dir, "pred_" + image_id+".json")):
        anno_helper = train_utils.AnnoUtil(os.path.join(annos_dir, "pred_" + image_id+".json"))
        image_annotations = anno_helper.image_annotations
        is_labeled = False
    else:
        print(f"Could not find annotations for {image_id}")
        return
    # Create tissue list and populate with tissue boxes
    annos = []
    # QuPath sometimes returns a dict instead of list of dicts
    if type(image_annotations) == dict:
        image_annotations = [image_annotations]
    for i in image_annotations:
        # Get box name, check if valid box
        box_name = anno_helper.get_box_name(i)
        if not box_name:
            continue
        # check partial match
        if not any(t.lower() in box_name.lower() for t in valid_tissues):
            continue
        
        # If we reach here, it means 'box_name' contains at least one valid tissue keyword
        box_dict = anno_helper.find_bbox_coordinates(i)
        annos.append(box_dict)
    #Create lesion list and populate with polygons
    if is_labeled:
        lesions = []
        for i in image_annotations:
            # Check if polygon
            num_vertices = len(next(anno_helper.search_recursive(i, "coordinates"))[0])
            if num_vertices < 6: #TODO: find a better way to get only boxes
                # Get polygon name
                properties = next(anno_helper.search_recursive(i, "properties"))
                try:
                    poly_name = properties["name"].lower()
                   # print(poly_name)
                except:
                    continue
                if label in poly_name: #TODO: replace label with cat_map
                    # Get polygon coords
                    poly_coords = next(anno_helper.search_recursive(i, "coordinates"))[0]
                    # Get polygon dict
                    poly_dict = {poly_name: poly_coords}
                    lesions.append(poly_dict)
                else:
                    continue
            else:
                continue
        #Execute processing function
        class_conversion_file = '/home/chao_lab/SynologyDrive/chaolab_AI_path/unbiased_teacher2/configs/class_conversions/neoplastic.json'
        
        with open(class_conversion_file) as f:
            class_conv = json.load(f)
        process_tissue_polygons(class_conv, image_file, annos, lesions, max_dim, base_dim, out_dir, use_tiff, info_dict, is_human_labeled)
    else:
        process_unlabeled(image_file, annos, max_dim, out_dir, base_dim, info_dict, is_human_labeled)
    return print(f"Finished processing {image_id}")


def  tissue_finder_gt( src_dir: str, out_dir: str, image_path: str, max_dim: int, annos_dir: str, use_tiff: bool, is_human_labeled: bool, valid_tissues=None):
    if valid_tissues is None:
        valid_tissues = []

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    image_id = os.path.basename(image_path).split('.')[0]
    image_ext = '.' + image_path.split('.')[-1]
    image_file = os.path.join(src_dir, image_id + image_ext)

    # --- Load WSI info (OpenSlide or TiffFile) ---
    if not use_tiff:
        reader = OpenSlide(image_file)
        info_dict = convert_openslide_to_dict(reader)
        dimension_index, downsampling_factor = dimensions_index(info_dict)
        base_dim = info_dict['slide_dimensions'][::-1]  # H, W
    else:
        with tf.TiffFile(image_file) as slide:
            try:
                base_dim = slide.series[0].levels[0].shape
                base_dim = train_utils.channel_last(base_dim)
            except:
                image = tf.imread(image_file)
                base_dim = train_utils.channel_last(image.shape)
        info_dict = {}

    # --- Read tissue annotations (no polygons needed) ---
    annotation_file = os.path.join(annos_dir, image_id + ".json")
    pred_annotation_file = os.path.join(annos_dir, "pred_" + image_id + ".json")
    if os.path.exists(annotation_file):
        anno_helper = train_utils.AnnoUtil(annotation_file)
        image_annotations = anno_helper.image_annotations
    elif os.path.exists(pred_annotation_file):
        anno_helper = train_utils.AnnoUtil(pred_annotation_file)
        image_annotations = anno_helper.image_annotations
    else:
        print(f"Could not find annotations for {image_id}")
        return

    # Possibly unify QuPath dict vs list
    if isinstance(image_annotations, dict):
        image_annotations = [image_annotations]

    # Collect all tissue bounding boxes
    tissue_boxes = []
    for ann in image_annotations:
        box_name = anno_helper.get_box_name(ann)
        if not box_name:
            continue
        if any(t.lower() in box_name.lower() for t in valid_tissues):
            bbox_coords = anno_helper.find_bbox_coordinates(ann)
            tissue_boxes.append(bbox_coords)

    # If no tissues found, you can skip or handle differently
    if not tissue_boxes:
        print(f"No tissues found for {image_id}")
        return

    # --- Read the entire WSI at a suitable pyramid level that doesn't exceed max_dim ---
    # (Essentially pick the level where H or W is just under max_dim.)
    if not use_tiff:
        # Using OpenSlide
        # You can replicate the logic you have to find the level that ensures
        # (downsampled_H <= max_dim) and (downsampled_W <= max_dim)
        # then read_region(...) at that level
        target_level = pick_level_that_fits(info_dict, max_dim)
        region_size = info_dict['level_dimensions'][target_level]
        full_wsi_image = reader.read_region(
            location=(0, 0), 
            level=target_level, 
            size=region_size
        ).convert('RGB')
        full_wsi_image = np.array(full_wsi_image)
        levelH, levelW = full_wsi_image.shape
    else:
        # Using TiffFile
        with tf.TiffFile(image_file) as slide:
            pyramid_reader = slide.series[0].levels  # or slide.pages if single-level
            target_level = pick_level_that_fits_tiff(pyramid_reader, base_dim, max_dim)
            full_wsi_image = pyramid_reader[target_level].asarray()
            full_wsi_image = train_utils.channel_last(full_wsi_image)
            levelH, levelW, _ = full_wsi_image.shape

    # If even the chosen level is still bigger than max_dim, do an extra resizing
    if full_wsi_image.shape[0] > max_dim or full_wsi_image.shape[1] > max_dim:
        scale_factor = max(
            full_wsi_image.shape[0] / max_dim, 
            full_wsi_image.shape[1] / max_dim
        )
        new_h = int(full_wsi_image.shape[0] / scale_factor)
        new_w = int(full_wsi_image.shape[1] / scale_factor)
        full_wsi_image = train_utils.resize_image(full_wsi_image, max_dim)
    else:
        scale_factor = 1.0

    # 1) Scale factor from base level 0 to target_level
    level_scale_factor = get_level_scale_factor(info_dict, target_level)

    # 2) Suppose we also had to resize that target level to fit in max_dim
    #    If we ended up with final_wsi_image of shape (newH, newW),
    #    but the pyramid level had shape (levelH, levelW), we can do:
    resize_scale_factor_h = full_wsi_image.shape[0] / levelH
    resize_scale_factor_w = full_wsi_image.shape[1] / levelW
    # Typically choose one or average them if they differ
    resize_scale_factor = (resize_scale_factor_h + resize_scale_factor_w) / 2.0
    ###### i can also use newh and neww
    composite_scale_factor = level_scale_factor * resize_scale_factor
    # Now create a single JSON annotation with bounding boxes for each tissue:
    # 1. We scale the bounding boxes by the same factor used to get full_wsi_image.
    # 2. Then store them in one dictionary.
    scaled_tissue_annos = []
    for box in tissue_boxes:
        for k, coords in box.items():
            x1, y1, x2, y2 = coords
            
            # Scale from base-level coordinates to final displayed coordinates:
            final_x1 = int(x1 * composite_scale_factor)
            final_y1 = int(y1 * composite_scale_factor)
            final_x2 = int(x2 * composite_scale_factor)
            final_y2 = int(y2 * composite_scale_factor)

            # Build annotation:
            scaled_tissue_annos.append({
                "category_id": 0,   # or anything appropriate ig
                "bbox": [final_x1, final_y1, final_x2, final_y2],
                "bbox_mode": 0,})

    # Construct the single final annotation dictionar
    tissue_finder_annotation = {
        "file_name": f"{image_id}.npy",
        "image_id": image_id,
        "original_width": base_dim[1],
        "original_height": base_dim[0],
        "width": full_wsi_image.shape[1],
        "height": full_wsi_image.shape[0],
        "tissues": scaled_tissue_annos,
        "labeled": str(is_human_labeled)
    }

    # --- Save outputs ---
    np.save(os.path.join(out_dir, f"{image_id}.npy"), full_wsi_image)
    with open(os.path.join(out_dir, 'tissue_annotations', f"{image_id}.json"), 'w') as f:
        json.dump(tissue_finder_annotation, f, indent=4, cls=NpEncoder)


# Define main function to get the inputs and run the function
if __name__ == '__main__':
    # Parse the arguments in_dir, out_dir, qupath anno folder, name (default neoplastic)
    parser = argparse.ArgumentParser(description='Get the input arguments')
    parser.add_argument(
        'src_dir', 
        type=str, 
        help='The source directory containing the images'
    )

    parser.add_argument(
        'out_dir', 
        type=str, 
        help='The output directory to save the processed images'
    )

# making the below as optional
    parser.add_argument(
        '--qupath_annos',
        type=str,
        default=None,
        help='(Optional) The directory containing the QuPath annotations. '
         'If not provided, auto-detection will be used.'
    )

    # 1B: add new argument for tissue types JSON
    parser.add_argument(
        '--tissue_json',
        type=str, 
        default='configs/class_conversions/tissues.json',
        help='Path to a JSON file containing valid tissue types. Default is '
            'configs/class_conversions/tissues.json.'
    )

    # issue 2
    parser.add_argument(
        '--mode',
        type=str,
        default='lesion_finder',
        choices=['lesion_finder', 'tissue_finder'],
        help='Mode to run the script: "lesion_finder" for the original cropping approach, '
            '"tissue_finder" for a single WSI image with tissue boxes.'
    )

    parser.add_argument(
        '--label',
        type=str,
        default='neoplastic',
        help='The label to use for the annotations, default is neoplastic'
    )

    # add optional argument to support tissue masking
    parser.add_argument(
        '--tissue_mask',
        type=bool,
        default=False,
        help='Boolean to indicate if tissue masking is needed'
    )

    parser.add_argument(
        '--cat_map',
        type=str,
        default='',
        help='Path to a cat_map JSON for category mapping. If empty, defaults are used.'
    )
    # Parse the arguments
    args = parser.parse_args()
    src_dir = args.src_dir
    out_dir = args.out_dir
    if args.qupath_annos != None:
        annos_dir = args.qupath_annos
    else:
        annos_dir = auto_detect_annotation_dir(src_dir)
        if annos_dir is None: # incase we still dont find anything
            print("No valid annotation directory was selected. Exiting.")
            exit()
    is_human_labeled = False
    if 'qupath_annotations_latest' in annos_dir:
        is_human_labeled = True

    # below part is for the json # 1B
    if not os.path.exists(args.tissue_json):
        print(f"WARNING: Tissue JSON not found at {args.tissue_json}. Using fallback list.")
        valid_tissues = ["lymph", "tissuefinder"]
    else:
        with open(args.tissue_json, 'r') as f:
            tissue_data = json.load(f)
        # we assume the key is "valid_tissues"
        valid_tissues = tissue_data.get("valid_tissues", [])
        if not valid_tissues:
            print("WARNING: 'valid_tissues' list is empty. No tissue boxes will be processed.")

    mode = args.mode

    label = args.label.lower()
    
    use_tiff = False # temp thing i added. TODO: make this more solid

    # Run the lesion_finder_gt function 
    for image_path in glob.glob(os.path.join(src_dir, '*.svs')): # tif or svs #TODO
        if mode == 'lesion_finder':
            lesion_finder_gt(src_dir, out_dir, image_path, 2560, annos_dir, use_tiff, is_human_labeled, valid_tissues=valid_tissues, label=label)
        elif mode == 'tissue_finder':
            tissue_finder_gt(src_dir, out_dir, image_path, max_dim=2560, annos_dir=annos_dir, use_tiff=use_tiff, is_human_labeled=is_human_labeled, valid_tissues=valid_tissues
        )
        else:
            print(f"Unknown mode: {mode}")
        
    
    print("Finished processing all images")
    
    