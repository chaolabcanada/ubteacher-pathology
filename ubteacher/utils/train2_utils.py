import os
import numpy as np
from typing import Dict, Tuple, List, Set, Iterator
import json
import glob
from detectron2.data import transforms as T
import tifffile as tf

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
        
def find_img_dir(parent_dir: str) -> List[str]:
    """
    Find npy image directory for training
    """
    img_dirs = []
    for root, dirs, files in os.walk(parent_dir):
        for d in dirs:
            if glob.glob(os.path.join(root, d, '*.npy')):
                img_dirs.append(os.path.join(root, d))
    # user chooses if there are multiple img folders
    for i, img_dir in enumerate(img_dirs):
        print(f'{i}: {os.path.relpath(img_dir, parent_dir)}')
    choice = input('Choose image folder index')
    if choice.isdigit() and int(choice) < len(img_dirs):
        return img_dirs[int(choice)]
    else:
        raise ValueError('Image folder not found')
        
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

def parse_json_by_task(json_file):
    """
    Parse json file by task
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
        return data
    
def get_scaling(original_file, output_file):
        with tf.TiffFile(original_file) as tiff:
            # get base size
            base_dim = tiff.pages[0].shape[:2]
            print(f'Image size: {base_dim}') #TODO: remove
    
        f = np.load(output_file)
        target_dim = f.shape[:2]
        del f # use del instead of with because numpy version issue
        print(f'Image size: {target_dim}') #TODO: remove
        return base_dim, target_dim

class ParseFromQuPath:
    
    def __init__(self, ref_dim, target_dim, tissue_types):
        self.anno_dir = '/mnt/RSX/Datasets_pathology/SRI_OSCC_lymph_labeled/qupath_annotations_latest'
        self.img_dir = '/mnt/RSX/Datasets_pathology/GT_2023/TissueFinderV2/SRI_OSCC'
        self.tissue_types = tissue_types
        self.ref_dim = ref_dim
        self.target_dim = target_dim
            
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
            # make it as 'bbox': asdjhkf
        return anno
        
    def get_boxes(self, json_file):
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        tissue_data = []
        for i in data:
            if any(tissue in list(search_recursive(i, 'name')) for tissue in self.tissue_types):
                tissue_data.append(i)
        cat_map = {tissue: i for i, tissue in enumerate(self.tissue_types)}
        coords = []
        for k in tissue_data:
            ## add names to k 
            k['geometry']['category_id'] = cat_map[next(search_recursive(k, 'name'))]
            del k['geometry']['type']
            k['geometry']['bbox_mode'] = 0
            coords.append(next(search_recursive(k, 'geometry')))
        
        out = self.scale_bboxes_qupath(coords)
        
        return out, cat_map

    def get_coco_format(self, json_file):
        
        """
        Get coco format for detectron2
        """
        ## Determine image format
        fmt = os.path.splitext(os.listdir(self.img_dir)[0])[1]
        img_base = os.path.basename(os.path.splitext(json_file)[0])
        img_fname = os.path.join(self.img_dir, img_base) + '.npy'
        
        ## Get annotation data
        
        annotation_dicts, cat_map = self.get_boxes(json_file)
        
        ## Fill remaining fields
        
        dataset_dicts = [{'file_name': img_fname,
                        'height': self.target_dim[0],
                        'width': self.target_dim[1],
                        'image_id': img_base,
                        'annotations': annotation_dicts}
                        ]  

        return dataset_dicts, cat_map