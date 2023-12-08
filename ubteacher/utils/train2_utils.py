import os
import numpy as np
from typing import Dict, Tuple, List, Set, Iterator
import json
import glob
from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog
import tifffile as tf
import random
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

def find_unlabeled_dirs(img_parent: str) -> List[str]:
    img_dirs = []
    for root, dirs, files in os.walk(img_parent):
        for d in dirs:
            if glob.glob(os.path.join(root, d, '*.npy')):
                img_dirs.append(os.path.join(root, d))
    # user chooses if there are multiple img folders
    for i, img_dir in enumerate(img_dirs):
        print(f'{i}: {os.path.relpath(img_dir, img_parent)}')
    choice = input('Choose image folders indices, comma separated: \n')
    choice = [int(i) for i in choice.split(',')]
    total_imgs = 0
    for folder in choice:
        total_imgs += len(glob.glob(os.path.join(img_dirs[folder], '*.npy')))
    print(f'Found {total_imgs} images')
    img_dirs = [img_dirs[i] for i in choice]
    
    return img_dirs
        
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
    choice = input('Choose image folders indices, comma separated: \n')
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
        print(anno_dir)
        if os.path.exists(anno_dir):
            total_annos.append(anno_dir)
        else:
            raise ValueError(f'Annotation folder not found for {base_dir}')
    
    return img_dirs, total_annos

        
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
    
    def __init__(self, anno_dir, img_dir, ref_dim, target_dim, tissue_types):
        self.anno_dir = anno_dir
        self.img_dir = img_dir
        self.ref_dim = ref_dim
        self.target_dim = target_dim
        self.tissue_types = tissue_types
            
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
        
        return out

    def split_dataset(self, cfg, dataset_names: List[str]):
        """Function to collect all input datasets and split them
        into 'train' and 'val' sets.
        Args:
        dataset_names: a list of dataset names in the dataset directory
        args: parsed command-line arguments
        
        """
        data_train = dict()
        data_val = dict()
        parent_dir = cfg.PARENTDIR
        set_seed = cfg.SET_SEED

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
    
    def get_coco_format(self, json_file):
        
        """
        Get labeled coco format for detectron2
        """
        ## Determine image format
        img_base = os.path.basename(os.path.splitext(json_file)[0])
        img_fname = os.path.join(self.img_dir, img_base) + '.npy'
        
        ## Get annotation data
        
        annotation_dicts = self.get_boxes(json_file)
        
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
                json.dump(data, f)
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
