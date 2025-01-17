import glob
import json
import os
import argparse
from typing import Dict, Tuple, List, Set, Iterator
import multiprocessing as mp
import numpy as np
import tifffile as tf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from matplotlib import patches
from detectron2.data import transforms as T
from utils.train_utils import AnnoUtil, TrainUtil, channel_last, scale_bboxes, resize_image
import utils.train_utils as train_utils

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
- Add support for other image formats
- Add support for tissue masking
- Automatically find the annotation folder (default: qupath_annotations_latest)
- Automatically find the TF prediction folder (default: tissue_finder_xxxx)
- Add support for cat map -> we want to load the cat_map and assign classes in "process_tissue_polygons"
- Support if class conversions are not provided
- Create better defaults
- Create better arg parse (i.e. help, required args replacing optional etc.)
- Refactor for readability, functionality is fine
- Macenko normalization

- Multi-process i/o function
'''

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


def process_unlabeled(image_file, annos, max_dim, out_dir, base_dim):
    image_id = os.path.basename(image_file).split(".")[0]
    with tf.TiffFile(image_file) as slide:
        for n, a in enumerate(annos):
            # Calculate cropped tissue dimensions at each level of the image pyramid
            for k, v in a.items():
                tissue_coords_base = v
                tx1, ty1, tx2, ty2 = tissue_coords_base
            # print("    ", f"annotation is '{k}'")
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
                    "labeled" : "False"
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
        
    
def process_tissue_polygons(class_conv, image_file, annos, lesions, max_dim, base_dim, out_dir):
    image_id = os.path.basename(image_file).split(".")[0]
    with tf.TiffFile(image_file) as slide:
        # Process each tissue
        for n, a in enumerate(annos):
            # Calculate cropped tissue dimensions at each level of the image pyramid
            for k, v in a.items():
                tissue_coords_base = v
                tx1, ty1, tx2, ty2 = tissue_coords_base
                base_long_edge = max(tx2-tx1, ty2-ty1)
            # print("    ", f"annotation is '{k}'")
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
                crop_w = cx2-cx1
                crop_h = cy2-cy1
                if crop_h>max_dim or crop_w>max_dim:
                    target_level = level_idx
                    continue
                else:
                    target_level = level_idx - 1
                    break
            # Read image at the correct level
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
            new_long_edge = max(recropped_tissue.shape[0], recropped_tissue.shape[1])
            scale_factor = round(new_long_edge/ base_long_edge, 3)
            
            if not os.path.exists(os.path.join(out_dir, 'visualizations')):
                os.makedirs(os.path.join(out_dir, 'visualizations'))
                        
            # Visualize
            fig, ax = plt.subplots()
            ax.imshow(recropped_tissue)  
            plt.savefig(os.path.join(out_dir, 'visualizations', f'{image_id}_{n}.png'))
            plt.close()

             # For each tissue, save to disk with annotations in json
            np.save(os.path.join(out_dir, f"{image_id}_{n}.npy"), recropped_tissue)
            
            annotation_dicts = []
            
            # Scale and offset lesion annotations (polygons) to tissue
            #print("    ", "Resizing lesion annotations (polygons)...")
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
                each_anno = {
                    "category_id" : 0, # TODO: Change this based on above 216-225
                    "bbox" : poly_bbox,
                    "bbox_mode" : 0,
                   # "segmentation" : [poly]
                }
                annotation_dicts.append(each_anno) 
            
            # At the end of this: we want to save a "cat_map" -> json file which maps the class names to the class numbers
            # {0: "non-neoplastic", 1: "neoplastic", 2: "ignore"}  
                
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
                "labeled" : "True"
            }
            if not os.path.exists(os.path.join(out_dir, 'tissue_annotations')):
                os.makedirs(os.path.join(out_dir, 'tissue_annotations'))
            anno_json = json.dumps(tissue_anno_dict, indent = 4, cls=NpEncoder)
            with open(os.path.join(out_dir, 'tissue_annotations', f"{image_id}_{n}.json"), 'w') as j:
                j.write(anno_json)
    return

def lesion_finder_gt(src_dir, out_dir, image_path, max_dim, annos_dir, label='neoplastic'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    image_id = image_path.split('/')[-1].split('.')[0]
    image_ext = '.' + image_path.split('/')[-1].split('.')[1]
    image_file = os.path.join(src_dir, image_id+image_ext)
        
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
        if box_name:
            if "prostate" in box_name or "tissuefinder" in box_name: # TODO: Add this as an input
                # Get box dict
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
        process_tissue_polygons(class_conv, image_file, annos, lesions, max_dim, base_dim, out_dir)
    else:
        process_unlabeled(image_file, annos, max_dim, out_dir, base_dim)
    return print(f"Finished processing {image_id}")


# Define main function to get the inputs and run the function
if __name__ == '__main__':
    # Parse the arguments in_dir, out_dir, qupath anno folder, name (default neoplastic)
    parser = argparse.ArgumentParser(description='Get the input arguments')
    parser.add_argument(
        '--src_dir', 
        type=str, 
        help='The source directory containing the images'
        )
    parser.add_argument(
        '--out_dir', 
        type=str, 
        help='The output directory to save the processed images'
        )
    parser.add_argument(
        '--qupath_annos',
        type=str,
        help='The directory containing the QuPath annotations'
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
    # Parse the arguments
    args = parser.parse_args()
    src_dir = args.src_dir
    out_dir = args.out_dir
    annos_dir = args.qupath_annos
    label = args.label.lower()
    
    # Run the lesion_finder_gt function 
    for image_path in glob.glob(os.path.join(src_dir, '*.svs')): # tif or svs #TODO
        try:
            lesion_finder_gt(src_dir, out_dir, image_path, 2560, annos_dir, label)
        except:
            print(f"Could not process {image_path}")
        
    ## TODO: Support image formats other than .svs
    
    print("Finished processing all images")
    
    