import os
import numpy as np
from typing import Dict, Tuple, List, Set, Iterator
import json
from detectron2.data import transforms as T

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
    
def scale_bboxes_qupath(anno, ref_dim, target_dim):
    x_scale = ref_dim[1] / target_dim[1]
    y_scale = ref_dim[0] / target_dim[0]
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
    
def get_boxes(json_file, ref_dim, target_dim, tissue_types):
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    tissue_data = []
    for i in data:
        if any(tissue in list(search_recursive(i, 'name')) for tissue in tissue_types):
            tissue_data.append(i)
    cat_map = {tissue: i for i, tissue in enumerate(tissue_types)}
    coords = []
    for k in tissue_data:
        ## add names to k 
        k['geometry']['category_id'] = cat_map[next(search_recursive(k, 'name'))]
        del k['geometry']['type']
        k['geometry']['bbox_mode'] = 0
        coords.append(next(search_recursive(k, 'geometry')))
    
    out = scale_bboxes_qupath(coords, ref_dim, target_dim)
    return coords, cat_map

def resize_and_limit(image, min_dim, max_dim):
    """
    Resize by minimum dimension unless it exceeds maximum
    Args:
    image -- np.ndarray with shape (h, w, c)
    Return:
    resized_image -- np.ndarray with shape (h, w, c)
    """
    h, w = image.shape[:2]
    base_dim = (h,w)
    scale_factor = round((min_dim / min(h, w)), 2)
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    if max(new_h, new_w) > max_dim:
        scale_factor = round((max_dim / max(new_h, new_w)), 2)
        new_h = int(new_h * scale_factor)
        new_w = int(new_w * scale_factor)
    target_dim = (new_h, new_w)
    resized_image = T.ResizeTransform(h, w, new_h, new_w).apply_image(image)
    return resized_image, base_dim, target_dim

def get_coco_format(parent_dir, max_dim, min_dim, img_dir, json_file, tissue_types):
    
    """
    Get coco format for detectron2
    """
    ## Determine image format
    fmt = os.path.splitext(os.listdir(parent_dir)[0])[1]
    img_base = os.path.basename(os.path.splitext(json_file)[0])
    img_fname = os.path.join(img_dir, img_base) + '.npy'
  
    #Scale and get dimensions
    img, ref_dim, target_dim = resize_and_limit(np.load(img_fname), min_dim, max_dim)
    
    ## Get annotation data
    
    annotation_dicts, cat_map = get_boxes(json_file, ref_dim, target_dim, tissue_types)
    
    ## Fill remaining fields
    
    dataset_dicts = [{'file_name': img_fname,
                      'height': ref_dim[0],
                      'width': ref_dim[1],
                      'image_id': img_base,
                      'annotations': annotation_dicts}
                    ]  

    return img, dataset_dicts, cat_map