import os
import numpy as np
from typing import Dict, Tuple, List, Set, Iterator
import json
import glob
from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog
import tifffile as tf
import random
import copy
import cv2
import argparse

def find_file(parent_dir, key, fmts):
    for root, dirs, files in os.walk(parent_dir):
        for fmt in fmts:
            for f in files:
                if f.startswith(key) and f.endswith(fmt):
                    return os.path.join(root, f)
        raise ValueError(f'File not found for {key}')

def select_annotypes(anno_dirs: str) -> List[str]:
    """
    Select annotation types to include
    """
    annotypes = []
    possible_tissues = []
    for anno_dir in anno_dirs:
        for f in glob.glob(os.path.join(anno_dir, '*.json')):
            with open(f, 'r') as f:
                data = json.load(f)
            for i in data:
                try:
                    if i['geometry']['type'] == 'Polygon':
                        possible_tissues += [next(search_recursive(i, 'name'))]
                        possible_tissues = list(set(t.split(' ')[0] for t in possible_tissues))
                except:
                    pass
    print(f'Found {set(possible_tissues)} tissue types with valid annotations')
    selected_tissues = input('Select tissue types to train on (comma separated): \n')
    tissue_types = selected_tissues.split(', ')
    print(f'Selected tissue types: {tissue_types}')
    annotypes.extend(tissue_types)
    return annotypes

def select_convert_annotypes(anno_dirs: str, class_file: Dict) -> List[str]:
    annotypes = []
    with open(class_file, 'r') as f:
        class_data = json.load(f)
    possible_conversions = list(class_data.keys())
    conversion_lengths = [len(class_data[i]) for i in possible_conversions]
    convertible_info = [f'{i}: {j}' for i, j in zip(possible_conversions, conversion_lengths)]
    print('Possible conversions: \n' + '\n'.join(convertible_info))
    selected_conversion = input('Select classes to train on (comma separated): \n')
    class_types = selected_conversion.split(', ')
    print(f'Selected classes: {class_types}')
    annotypes.extend(class_types) 
    return annotypes 

def find_unlabeled_dirs(img_parent: str) -> List[str]:
    img_dirs = []
    for root, dirs, files in os.walk(img_parent):
        for d in dirs:
            if glob.glob(os.path.join(root, d, '*.npy')):
                img_dirs.append(os.path.join(root, d))
    # user chooses if there are multiple img folders
    for i, img_dir in enumerate(img_dirs):
        print(f'{i}: {os.path.relpath(img_dir, img_parent)}')
    choice = input('Choose unlabeled image folders indices, comma separated: \n')
    choice = [int(i) for i in choice.split(',')]
    total_imgs = 0
    for folder in choice:
        total_imgs += len(glob.glob(os.path.join(img_dirs[folder], '*.npy')))
    print(f'Found {total_imgs} images')
    img_dirs = [img_dirs[i] for i in choice]
    
    return img_dirs

def flip_dict_and_filter(d: Dict, c: List) -> Dict:
    inverse = {}
    for k,v in d.items():
        for x in v:
            if not x in inverse:
                    inverse.setdefault(x, []).append(k)
    #flatten values
    inverse = {k: v[0] for k, v in inverse.items()}
    #filter values for only those in c
    inverse = {k: v for k, v in inverse.items() if k in c}
    return inverse
        
def find_dirs(anno_parent: str, img_parent: str) -> List[str]:
    """
    Find all annotation and image folders
    """       
    
    img_dirs = []
    for root, dirs, files in os.walk(img_parent):
        for d in dirs:
            if glob.glob(os.path.join(root, d, '*.npy')):
                img_dirs.append(os.path.join(root, d))
    # user chooses if there are multiple img folders
    for i, img_dir in enumerate(img_dirs):
        print(f'{i}: {os.path.relpath(img_dir, img_parent)}')
    choice = input('Choose labeled image folders indices, comma separated: \n')
    choice = [int(i) for i in choice.split(',')]
    total_imgs = 0
    for folder in choice:
        total_imgs += len(glob.glob(os.path.join(img_dirs[folder], '*.npy')))
    print(f'Found {total_imgs} images')
    img_dirs = [img_dirs[i] for i in choice]
    
    #See if qupath_annotations_latest exists in an img_dir
    
    sample_dir = os.path.join(anno_parent, os.path.basename(img_dirs[0]))
    if os.path.exists(os.path.join(sample_dir, 'qupath_annotations_latest')):
        print('Automatically chose qupath_annotations_latest folder')
        anno_subdir = ('qupath_annotations_latest')
    else:
        anno_dirs = []
        for root, dirs, files in os.walk(sample_dir):
            for d in dirs:
                if 'annotations' in d:
                    anno_dirs.append(os.path.join(root, d))
        # user chooses if there are multiple annotation folders
        print('Found multiple annotation folders:')
        for i, anno_dir in enumerate(anno_dirs):
            print(f'{i}: {os.path.relpath(anno_dir, anno_parent)}')
        choice = input('Choose annotation folder index: \n')
        if choice.isdigit() and int(choice) < len(anno_dirs):
            anno_subdir = os.path.basename(anno_dirs[int(choice)])
        else:
            raise ValueError('Annotation folder not found')
    
    total_annos = []
    for img_dir in img_dirs:
        base_dir = os.path.basename(img_dir)
        anno_dir = os.path.join(anno_parent, base_dir, anno_subdir)
        total_annos.append(anno_dir)
    return total_annos, img_dirs

        
def search_recursive(d: Dict, key: str) -> Iterator:
        """Helper function for finding which level of json annotations has
        the matching key.
        """
        for k, v in d.items():
            if isinstance(v, Dict):
                for match in search_recursive(v, key):
                    yield match
            if k == key:
                # generator function - saves in memory until called
                # (use for loop to call)
                yield v

def search_with_targets(d: Dict, key: str, targets: List[str]) -> Iterator:
    for k, v in d.items():
        if isinstance(v, Dict):
            for match in search_with_targets(v, key, targets):
                yield match
        if k == key and v in targets:
            yield v
    
def get_scaling(original_file, output_file):
        with tf.TiffFile(original_file) as tiff:
            # get base size
            try:
                base_dim = tiff.pages[0].shape[:2]
            except:
                base_dim = tiff.series[0].shape[:2]     
        f = np.load(output_file)
        target_dim = f.shape[:2]
        del f # use del instead of with because numpy version issue
        return base_dim, target_dim
    
class ParseUnlabeled:
    def __init__(self, u_img_dir):
        self.u_img_dir = u_img_dir
        
    def get_unlabeled_coco(self, img_file):
            
            """
            Get unlabeled coco format for detectron2
            """
            f = np.load(img_file)
            shape = f.shape
            del f # use del instead of with because numpy version issue
            img_base = os.path.basename(os.path.splitext(img_file)[0])
                       
            ## Fill remaining fields
            
            dataset_dicts = [{'file_name': img_file,
                            'height': shape[0],
                            'width': shape[1],
                            'image_id': img_base}
                            ]  
    
            return dataset_dicts 
    

class ParseFromQuPath:
    
    def __init__(self, anno_dir, img_dir, ref_dim, target_dim, 
                 class_types, class_file=None, box_only=True):
        
        self.anno_dir = anno_dir
        self.img_dir = img_dir
        self.ref_dim = ref_dim
        self.target_dim = target_dim
        self.class_types = class_types
        self.box_only = box_only
        self.class_file = class_file
        
        if self.class_file:
            self.qupath_classes = []
            with open(class_file, 'r') as f:
                self.class_data = json.load(f)
            for chosen in self.class_types:
                self.qupath_classes.append(self.class_data[chosen])
            self.qupath_classes = [item for sublist in self.qupath_classes for item in sublist]          
            
    def scale_bboxes_qupath(self, anno):
        x_scale = self.ref_dim[1] / self.target_dim[1]
        y_scale = self.ref_dim[0] / self.target_dim[0]
        for i in anno:
            [coords] = i['coordinates']
            # First, build XYXY
            x0 = int(coords[0][0] / x_scale)
            y0 = int(coords[0][1] / y_scale)
            x1 = int(coords[2][0] / x_scale)
            y1 = int(coords[2][1] / y_scale)
            i['bbox'] = [x0, y0, x1, y1]
            del i['coordinates']
        return anno
    
    def scale_polygons_qupath(self, anno):
        x_scale = self.ref_dim[1] / self.target_dim[1]
        y_scale = self.ref_dim[0] / self.target_dim[0]
        for i in anno:
            poly = []
            # Build XYXYXY...
            for xn, yn in i['coordinates'][0]:
                xn = int(xn / x_scale)
                yn = int(yn / y_scale)
                poly.append(xn); poly.append(yn)
            i['segmentation'] = [poly]
            [coords] = i['coordinates']
            # Build XYXY for bbox
            x0 = int(coords[0][0] / x_scale)
            y0 = int(coords[0][1] / y_scale)
            x1 = int(coords[2][0] / x_scale)
            y1 = int(coords[2][1] / y_scale)
            i['bbox'] = [x0, y0, x1, y1]
            del i['coordinates']
        return anno
    
    def scale_back_to_qupath(self, anno, mode='box'):
        x_scale = self.target_dim[1] / self.ref_dim[1]
        y_scale = self.target_dim[0] / self.ref_dim[0]
        if mode == 'box':
            for i in anno:
                [coords] = i['coordinates']
                # First, build XYXY
                x0 = int(coords[0][0] * x_scale)
                y0 = int(coords[0][1] * y_scale)
                x1 = int(coords[2][0] * x_scale)
                y1 = int(coords[2][1] * y_scale)
                i['bbox'] = [x0, y0, x1, y1]
                del i['coordinates']
        else:
            for i in anno:
                poly = []
                # Build XYXYXY...
                for xn, yn in i['coordinates'][0]:
                    xn = int(xn / x_scale)
                    yn = int(yn / y_scale)
                    poly.append(xn); poly.append(yn)
                i['segmentation'] = [poly]
                [coords] = i['coordinates']
                # Build XYXY for bbox
                x0 = int(coords[0][0] / x_scale)
                y0 = int(coords[0][1] / y_scale)
                x1 = int(coords[2][0] / x_scale)
                y1 = int(coords[2][1] / y_scale)
                i['bbox'] = [x0, y0, x1, y1]
                del i['coordinates']
        return anno
        
    def get_boxes(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        tissue_data = []
        for i in data:
            if not self.class_file:
                if any(classes in list(search_recursive(i, 'name')) for classes in self.class_types):
                    tissue_data.append(i)
        # create normal cat map in absence of conversion file      
        cat_map = {cat: i for i, cat in enumerate(self.class_types)}
        ## get coords
        coords = []        
        for k in tissue_data:
            ## add names to k 
            k['geometry']['category_id'] = cat_map[next(search_with_targets(k, 'name', self.class_types))]
            del k['geometry']['type']
            k['geometry']['bbox_mode'] = 0
            coords.append(next(search_recursive(k, 'geometry')))
        
        out = self.scale_bboxes_qupath(coords)
        return out
    
    def get_boxes_converted(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        tissue_data = []
    
        for i in data:
            if any(classes in list(search_recursive(i, 'name')) for classes in self.qupath_classes):
                tissue_data.append(i)
    
        ## create cat map with input classes instead of qupath classes
        flipped_map = flip_dict_and_filter(self.class_data, self.qupath_classes)
        cat_map_class_types = {cat: i for i, cat in enumerate(self.class_types)}
        cat_map = {k: cat_map_class_types[v] for k, v in flipped_map.items()}
        
        ## get coords
        coords = []        
        for k in tissue_data:
            ## add names to k 
            k['geometry']['category_id'] = cat_map[next(search_with_targets(k, 'name', self.qupath_classes))]
            del k['geometry']['type']
            k['geometry']['bbox_mode'] = 0
            if len(k['geometry']['coordinates'][0]) <= 5:
                coords.append(next(search_recursive(k, 'geometry')))
        out = self.scale_bboxes_qupath(coords)
        
        return out
        
    
    def get_polygons(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        tissue_data = []
        for i in data:
            if not self.class_file:
                if any(classes in list(search_recursive(i, 'name')) for classes in self.class_types):
                    tissue_data.append(i)
        # create normal cat map in absence of conversion file      
        cat_map = {cat: i for i, cat in enumerate(self.class_types)}
        
        coords = []        
        for k in tissue_data:
            ## add names to k 
            k['geometry']['category_id'] = cat_map[next(search_with_targets(k, 'name', self.class_types))]
            del k['geometry']['type']
            k['geometry']['bbox_mode'] = 0
            coords.append(next(search_recursive(k, 'geometry')))
        out = self.scale_polygons_qupath(coords)
        return out
        
    def get_polygons_converted(self, json_file):       
        with open(json_file, 'r') as f:
            data = json.load(f)
        tissue_data = []
    
        for i in data:
            try:
                if any(classes in list(search_recursive(i, 'name')) for classes in self.qupath_classes):
                    tissue_data.append(i)
            except:
                pass

        ## create cat map with input classes instead of qupath classes
        flipped_map = flip_dict_and_filter(self.class_data, self.qupath_classes)
        cat_map_class_types = {cat: i for i, cat in enumerate(self.class_types)}
        cat_map = {k: cat_map_class_types[v] for k, v in flipped_map.items()}
        
        ## get coords
        coords = []        
        for k in tissue_data:
            ## add names to k 
            k['geometry']['category_id'] = cat_map[next(search_with_targets(k, 'name', self.qupath_classes))]
            del k['geometry']['type']
            k['geometry']['bbox_mode'] = 0
            coords.append(next(search_recursive(k, 'geometry')))
        out = self.scale_polygons_qupath(coords)
        return out
    
    def get_coco_format(self, json_file):
        
        """
        Get labeled coco format for detectron2
        """
        ## Determine image format
        img_base = os.path.basename(os.path.splitext(json_file)[0])
        img_fname = os.path.join(self.img_dir, img_base) + '.npy'
        
        ## Get annotation data
        if self.box_only:
            if self.class_file:
                annotation_dicts = self.get_boxes_converted(json_file)
            else:
                annotation_dicts = self.get_boxes(json_file)
        else:
            if self.class_file:
                annotation_dicts = self.get_polygons_converted(json_file)
            else:
                annotation_dicts = self.get_polygons(json_file)
        
        ## Fill remaining fields
        
        dataset_dicts = [{'file_name': img_fname,
                        'height': self.target_dim[0],
                        'width': self.target_dim[1],
                        'image_id': img_base,
                        'annotations': annotation_dicts}
                        ]  

        return dataset_dicts
    
def split_dataset(cfg, dataset_dicts):
    """Function to split a dataset into 'train' and 'val' sets.
    Args:
    dataset_dicts: a list of dicts in detectron2 dataset format
    """
    # Try loading from data seed
    try:
        with open(cfg.DATASEED, 'r') as f:
            data = json.load(f)
            train_set = data['train']
            val_set = data['val']
            return train_set, val_set
    except:
        # Split dataset into train and val                
        random.seed(0)
        random.shuffle(dataset_dicts)
        split = int(len(dataset_dicts) * cfg.TRAIN_FRACTION)
        train_set = dataset_dicts[:split]
        val_set = dataset_dicts[split:]
        if cfg.SET_SEED:
            data = {'train': train_set, 'val': val_set}
            with open(cfg.DATASEED, 'w') as f:
                json.dump(data, f, indent=4)
        return train_set, val_set
        
    
def register_dataset(dset_type: str, dataset_dicts: Dict, classes: List[str]):
        """Helper function to register a new dataset to detectron2's
        Datasetcatalog and Metadatacatalog.

        Args:
        dataset_dicts -- list of dicts in detectron2 dataset format
        cat_map -- dictionary to map categories to ids, e.g. {'ROI':0, 'JUNK':1}
        """
        reg_name = dset_type
        
        # Register dataset to DatasetCatalog
        print(f"working on '{reg_name}'...")
        
        DatasetCatalog.register(
            reg_name,
            lambda d=dset_type: dataset_dicts
        )
        # Register metadata to MetadataCatalog
        MetadataCatalog.get(reg_name).set(
            thing_classes=classes
        )
        return MetadataCatalog


def merge_bboxes(box_coords: List[dict], iou_thresh: int=50) -> List[dict]:
    #TODO: add score and class as criteria for merging
    # Get boxes from list of dicts
    boxes = []
    for i in box_coords:
        boxes.append(i['bbox'])     
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
        reverse = True
        i = 0
        sorted_bboxes = sorted(boxes, key=lambda b:b[i], reverse=reverse)
    
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
            
    # Recreate instance_dicts with merged boxes
    box_arr2 = [{'bbox': i, 'score': 1} for i in box_arr2]
        
    return box_arr2

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
