"""
@Version: 0.4.5
@Author: Jesse Chao, PhD
@Contact: jesse.chao@sri.utoronto.ca
"""


import argparse
import re
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple, List, Set, Iterator
from functools import partial
import multiprocessing as mp

import cv2
import numpy as np
import tifffile as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from detectron2.data import transforms as T

from . import ROI_utils as train_utils


def clip_bbox(bbox, ref_width, ref_height):
    clipped = list(map(lambda p: max(p, 0), bbox))
    x1, y1, x2, y2 = clipped
    x2 = 0 if x2>ref_width else x2
    y2 = 0 if y2>ref_height else y2
    return [x1, y1, x2, y2]

def resize_image_cv(image:np.ndarray, max_dim=1500) -> np.ndarray:
    h, w = image.shape[:2]
    scale_factor = round((max_dim / max(h, w)), 2)
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    resized = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_AREA)
    return resized
    

class TissueFinder:
    def __init__(self, image: np.ndarray) -> None:
        """
        Args:
        image -- array with shape (h, w, c)
        """
        # Make sure image shape is channel-last
        if image.shape[0] == 3:
            image = np.transpose(image, [1, 2, 0])
        self.image = image
    
    @staticmethod
    def sort_boxes(bboxes: list, method="left-right") -> list:
        """Sort a list of bounding boxes in order of 'method'
        Args:
        bboxes -- a list of bboxes in format [XYWH]
        method -- one of 'left-right', 'right-left', 'top-bottom', 'bottom-top'
        Returns:
        a list of sorted boxes
        """
        if len(bboxes) < 2:
            return bboxes
        else:
            # Initialize the reverse flag and sort index
            reverse = False
            i = 0
            # If sorting in reverse
            if method == "right-left" or method == "bottom-top":
                reverse = True
            # If sorting against the y-coordinate rather than
            # the x-coordinate of the bounding box
            if method == "top-bottom" or method == "bottom-top":
                i = 1
            # Sort bounding boxes. Default: left-to-right
            sorted_bboxes = sorted(bboxes, key=lambda b:b[i], reverse=reverse)

            # return the list of sorted contours and bounding boxes
            return sorted_bboxes

    def merge_bboxes(self, boxes: list, iou_thresh: int=0) -> list:
        """Merge overlapping bounding boxes.
        Args:
        boxes -- list of boxes with format [x, y, w, h]
        iou_thresh -- threshold for IOU to trigger merging
        Return:
        a list of merged boxes without overlap
        """
        box_arr1 = self.sort_boxes(boxes)
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
                    x, y, w, h = box
                    boxes2compare.append(int(i) for i in [x, y, x+w, y+h])
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
                        merged_box = [A['x1'], min(A['y1'], B['y1']), u_w, u_h]
                        box_arr2.pop(-1)
                        box_arr2.append(merged_box)
                        # Look back
                        box_arr2 = self.merge_bboxes(box_arr2)
            # After looping through box_arr1, remove boxes that have been merged
            # or exhaustively compared from box_arr1
            box_arr1 = [i for n, i in enumerate(box_arr1) if n not in remove_indices]
            if not merge:  # If no merging event has occured
                # Add the 1st box from box_arr1 to box_arr2 for the next round
                box_arr2.append(box_arr1.pop(0))
        return box_arr2

    def boxes_from_contours(self, bin_mask: np.ndarray) -> np.ndarray:
        """Get bounding boxes from contours 
        and filter out ones that are too small
        i.e., likely not tissue"""
        # Get contours
        contours, hierarchy = cv2.findContours(
                                bin_mask,
                                mode=cv2.RETR_TREE,
                                method=cv2.CHAIN_APPROX_SIMPLE)
        # Get bounding boxes
        boxes_to_keep = []
        im_area = bin_mask.shape[0] * bin_mask.shape[1]
        for c in contours:
            cbox = cv2.boundingRect(c)
            x, y, w, h = cbox
            # Filter out boxes with weird ratios
            long_edge = max(w, h)
            short_edge = min(w, h)
            if long_edge/short_edge > 6:
                continue
            cbox_area = w * h
            # Filter out small boxes
            if im_area*0.99 > cbox_area > im_area*0.005:
                boxes_to_keep.append(cbox)
        return boxes_to_keep

    def expand_boxes(self, boxes:List, frac:float):
        """Expands boxes by a defined fraction of the original
        Args:
        boxes -- list of boxes in XYWH
        frac -- expansion factor between 0-1
        Returns:
        list of expanded boxes
        """
        bigger_boxes = []
        for b in boxes:
            x, y, w, h = b
            x1 = x - w*frac
            y1 = y - h*frac
            x2 = x + w + w*frac
            y2 = y + h + h*frac
            img_w = self.image.shape[1]
            img_h = self.image.shape[0]
            x1 = 0 if x1<0 else x1
            y1 = 0 if y1<0 else y1
            x2 = img_w if x2>img_w else x2
            y2 = img_h if y2>img_h else y2
            bigger_boxes.append([int(i) for i in [x1, y1, x2-x1, y2-y1]])
        return bigger_boxes

    def get_tissue_roi(self) -> List:
        """
        Get tissue ROIs in the form of bounding boxes
        from an whole slide image

        Return:
        list of boxes representing the tissues
        in (X, Y, W, H)
        """
        # Get binary mask
        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # Remove small holes
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
        # Get foreground segmentation mask
        fg_mask = cv2.erode(close, (3, 3), iterations=5)
        fg_mask = cv2.dilate(fg_mask, (3, 3), iterations=5)
        # Get bounding boxes
        rough_bboxes = self.boxes_from_contours(fg_mask)  
        # Merge overlapping boxes
        final_bboxes = self.merge_bboxes(rough_bboxes)
        # Expand boxes slightly
        expanded_bboxes = self.expand_boxes(final_bboxes, 0.1)
        return expanded_bboxes

    @staticmethod
    def get_anno_for_tissue(tissue_roi: List, annotations: List[Dict]):
        """
        Args:
        tissue_roi -- XYWH
        annotations -- a list of annotation dicts, which has keys
        
        Returns:
        list of annotations for each tissue
        """
        tissue_width, tissue_height = tissue_roi[2:]
        cropper = T.CropTransform(*tissue_roi)
        anno_per_tissue = []
        for anno in annotations:
            box = anno['bbox']
            cropped = cropper.apply_box(box)[0]
            clipped = clip_bbox(cropped, tissue_width, tissue_height)
            clipped = list(map(lambda p: max(p, 0), cropped))
            if clipped[-1]==0 or clipped[-2]==0:
                continue
            else:
                anno['bbox'] = clipped
                anno_per_tissue.append(anno)
        return anno_per_tissue

def get_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
    boxA -- ['x1', 'x2', 'y1', 'y2']
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    boxB -- ['x1', 'x2', 'y1', 'y2']
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns:
    iou -- float in [0, 1]
    """
    keys = ['x1', 'y1', 'x2', 'y2']
    bb1 = dict(zip(keys, boxA))
    bb2 = dict(zip(keys, boxB))
    
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def cleanup_annos(ref_image: np.ndarray, anno: List[Dict]) -> List[Dict]:
    """Remove smaller overlapping bboxes if present
    Args:
    anno -- a list of dicts whose key is the classification
            of the bbox, and value is bbox in [XYXY]
    
    Return:
    unique_annos -- similar list of dicts but overlapping
                    bboxes are merged if they have the same 
                    classification
    """
    box_keys = []
    for i in anno:
        box_keys.extend(i)

    unique_annos = []
    for bk in set(box_keys):
        boxes_to_merge = []
        for a in anno:
            for key, val in a.items():
                if key == bk:
                    boxes_to_merge.append(val)
        merged = TissueFinder(ref_image).merge_bboxes(boxes_to_merge)
        for m in merged:
            unique_annos.append({bk: m})
    return unique_annos


def get_valid_tissues(img_path: str, anno_path: str, max_dim: int) -> List:
    """
    Return only tissues with valid annotations
    """
    anno_helper = train_utils.AnnoUtil(anno_path)
    try:
        bbox_dicts = anno_helper.parse_bboxes()
    except KeyError:
        bbox_dicts = None
        print(f"Trouble parsing {anno_path}.")
        return None
    if not bbox_dicts:
        print(f"No valid annotations for {os.path.basename(img_path)}!")
        return None
    with tf.TiffFile(img_path) as slide:
        # Read image thumbnail
        top = train_utils.TrainUtil(max_dim).find_top(slide)
        top_dim = top.shape
        base_dim = slide.series[0].levels[0].shape
        base_dim = train_utils.channel_last(base_dim)
    # Detect large tissues
    tissues = TissueFinder(top).get_tissue_roi()
    # Scale bboxes
    try:
        scaled_bboxdicts = anno_helper.scale_bbox_dicts(bbox_dicts, base_dim, top_dim)
    except TypeError:
        print(f"TypeError: {img_path}")
        print(bbox_dicts, base_dim, top_dim)
        return None
    # Crop and clip annotations
    # Adjust verticies to the tissue boundary
    valid_tissues = []
    for t in tissues:
        cropper = T.CropTransform(*t)
        tx1, ty1, t_width, t_height = t
        tissue_box = [tx1, ty1, tx1+t_width, ty1+t_height]
        tissue_anno = []
        for i in scaled_bboxdicts:
            for name, box in i.items():
                try:
                    iou = get_iou(tissue_box, box)
                except AssertionError:
                    return None
                cropped = cropper.apply_box(box)[0]
                # Reposition out-of-bounds vertices
                clipped = list(map(lambda p: max(p, 0), cropped))
                bx1, by1, bx2, by2 = clipped
                if iou > 0:
                    bx2 = t_width if bx2>t_width else bx2
                    by2 = t_height if by2>t_height else by2
                else:
                    bx2 = 0 if bx2>t_width else bx2
                    by2 = 0 if by2>t_height else by2
                # Drop boxes that are clearly out-of-bounds
                if bx2==0 or by2==0:
                    continue
                # Drop boxes with strange dimensions
                box_w = bx2 - bx1
                box_h = by2 - by1
                box_ratio = max(box_w, box_h) / min(box_w, box_h)
                if box_ratio > 5:
                    continue
                tissue_anno.append({name: [bx1, by1, bx2, by2]})
        if tissue_anno:
            valid_tissues.append(
                {'tissue_box': tissue_box,
                    'tissue_anno': tissue_anno,
                    'ref_dim': top_dim})
    return valid_tissues

def vis_image_with_annos(image, annotations, output):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for anno in annotations:
        for name, box in anno.items():
            x1, y1, x2, y2 = box
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
            ax.annotate(name, (x1, y1), color='b')
        fig.savefig(output)
        plt.close()

def main(out_dir, img_file, anno_file, **kwargs):
    img_name = os.path.basename(img_file).split('.')[0]
    print(f"Processing {img_name}...")
    # Find matching annotation json
    max_dim = args.max_dim
    valid_tissues = get_valid_tissues(img_file, anno_file, max_dim)
    if valid_tissues:
        for n, t in enumerate(valid_tissues):
            tissue_box = t['tissue_box']
            tx1, ty1, tx2, ty2 = tissue_box
            tissue_dim = [ty2-ty1, tx2-tx1]
            tissue_anno = t['tissue_anno']
            ref_dim = t['ref_dim']
            # Calculate crop dimensions at each scale of the pyramid
            with tf.TiffFile(img_file) as tiff:
                if len(tiff.series[0].levels) > 1:
                # Use levels to load, most common
                    pyramid_reader = tiff.series[0].levels
                else:
                # Use pages to load, rare
                    pyramid_reader = tiff.pages
                for lvl_idx, level in enumerate(pyramid_reader):
                    lvl_dim = train_utils.channel_last(level.shape)
                    lvl_crop = train_utils.scale_bboxes([tissue_box], ref_dim, lvl_dim)[0]
                    cx1, cy1, cx2, cy2 = [int(i) for i in lvl_crop]
                    if (cx2-cx1)>max_dim or (cy2-cy1)>max_dim:
                        target = lvl_idx
                        continue
                    else:
                        target = lvl_idx - 1
                        break
                # Read image
                try:
                    im = pyramid_reader[target].asarray()
                except IndexError:
                    im = pyramid_reader[lvl_idx].asarray()
                im = train_utils.channel_last(im)
            # Crop image NOTE CropTransform(x0, y0, w, h)
            crop_coord = train_utils.scale_bboxes([tissue_box], ref_dim, im.shape)[0]
            tx1, ty1, tx2, ty2 = [int(i) for i in crop_coord]
            cropper = T.CropTransform(tx1, ty1, (tx2-tx1), (ty2-ty1))
            recropped_tissue = cropper.apply_image(im)
            if recropped_tissue.size:
                # Resize crop if necessary
                if recropped_tissue.shape[0] > max_dim or recropped_tissue.shape[1] > max_dim:
                    recropped_tissue = train_utils.resize_image(recropped_tissue, max_dim)
                tissue_anno = train_utils.AnnoUtil.scale_bbox_dicts(tissue_anno, tissue_dim, recropped_tissue.shape)
                # Save tissue image as numpy, annos as json
                tissue_fname = f"{img_name}_{n}"
                np.save(os.path.join(out_dir, tissue_fname), recropped_tissue)
                tissue_anno_dict = {
                    'file_name': img_file,
                    'pyramid_level': lvl_idx,
                    'tissue_xyxy': [tx1, ty1, tx2, ty2],
                    'image_width': recropped_tissue.shape[1],
                    'image_height': recropped_tissue.shape[0],
                    'max_dim': max_dim,
                    'box_dicts': tissue_anno
                    }
                try:
                    anno_json = json.dumps(tissue_anno_dict, indent=4)
                    with open(os.path.join(out_dir, 'tissue_annotations', tissue_fname+'.json'), 'w') as j:
                        j.write(anno_json)
                    if 'visualize' in kwargs.keys():
                        vis_file = tissue_fname+'.png'
                        vis_out_dir = os.path.join(out_dir, 'data_vis')
                        vis_image_with_annos(
                            recropped_tissue,
                            tissue_anno,
                            os.path.join(vis_out_dir, vis_file))
                except TypeError:
                    print(f"Cannot process {img_file}")
                    print(tissue_anno_dict)


if __name__ == '__main__':
    # ---------------------------------------
    # Setup commandline arguments
    # ---------------------------------------
    parser = argparse.ArgumentParser(description="Train an ROI detection model")
    parser.add_argument(
        "parent_dir",
        metavar="INPUT DIRECTORY",
        type=str,
        help="Parent directory to the datasets, ex: '/mnt/DQ8/SynologyDrive/'",
    )
    parser.add_argument(
        'output_dir',
        metavar='OUTPUT DIRECTORY',
        type=str,
        help="Output directory to save processed images and annotations, ex: 'tissue_finder_out"
    )
    parser.add_argument(
        "--max_dim",
        metavar="maximum dimension",
        type=int,
        nargs="?",
        default=1333,
        help="The maximum dimension for the longest edge of each input image. \
            Default=1333"
    )
    parser.add_argument(
        '--num_workers',
        metavar='multiprocessing_workers',
        type=int,
        nargs='?',
        default=int(mp.cpu_count()/2),
        help="Number of CPU threads for multiprocessing. \
            Default is half of total number of threads"
    )
    parser.add_argument(
        '--vis',
        action='store_true',
        help="visualize images with annotations"
    )

    # Parse commandline arguments
    print()
    args = parser.parse_args()
    output_dir = args.output_dir
    visualize = args.vis
    num_workers = args.num_workers
    if os.path.exists(output_dir):
        overwrite_ans = str(
            input("The output folder already exists, would you like to reuse it? Contents may be overwritten. [y/n]")
        )
        if "n" in overwrite_ans.lower():
            output_dir += "_v2"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'setup.json'), 'w') as f:
        f.write(json.dumps({'max_dim': args.max_dim}))
    compatible_formats = (
    "tif",
    "tiff",
    "stk",
    "lsm",
    "sgi",
    "gel",
    "svs",
    "scn",
    "sis",
    "bif",
    "zif",
    )
    potential_dsets = []
    for f in os.scandir(args.parent_dir):
        if f.is_dir() and not f.name.startswith('.'):
            potential_dsets.append(f.name)
    print(f"\nFound the following datasets: {sorted(potential_dsets)}")
    user_selection_dsets = [
        i.strip()
        for i in input(
            "\nEnter the names of datasets you want to process"
            + "(separate by comma):"
        ).split(",")
    ]
    for dset_name in user_selection_dsets:
        # Specify input dirs, collate images and annotations
        src_img_dir = os.path.join(args.parent_dir, dset_name)
        anno_dir = os.path.join(src_img_dir, "qupath_annotations_latest")
        img_files = []
        anno_files = []
        for file in os.scandir(src_img_dir):
            if not file.name.startswith(".") and file.name.endswith(compatible_formats):
                img_name = os.path.splitext(file.name)[0]
                anno_file = os.path.join(anno_dir, img_name + ".json")
                if not os.path.exists(anno_file):  # Only add images with matching annos
                    print(f"\nSkipping {img_name}; no annotation file found.")
                    continue
                else:
                    img_files.append(file.path)
                    anno_files.append(anno_file)
        print(f"\nFound {len(img_files)} source images in {dset_name}")
        
        # Specify output dirs, collect processed files (if exists)
        img_out_dir = os.path.join(output_dir, dset_name)
        anno_out_dir = os.path.join(img_out_dir, 'tissue_annotations')
        processed_files = set()
        if os.path.exists(img_out_dir):
            del_imgOutDir_ans = str(
                input(f"The destination folder for '{dset_name}' already exists. Would you like to remove it and start fresh? [y/n]"))
            if "y" in del_imgOutDir_ans.lower():
                shutil.rmtree(img_out_dir)
            else:
                regex = r"(_)[0-9](.npy)"
                for f in os.scandir(img_out_dir):
                    if f.name.endswith('.npy'):
                        file_name_root = f.name[:re.search(regex, f.name).span()[0]]
                        processed_files.add(file_name_root)
        else:
            os.makedirs(img_out_dir)
        os.makedirs(anno_out_dir, exist_ok=True)
        print(f"{len(processed_files)} already processed, {len(img_files)-len(processed_files)} new images to go.", end=" ")
        
        if visualize:
            vis_out_dir = os.path.join(img_out_dir, 'data_vis')
            os.makedirs(vis_out_dir, exist_ok=True)
        
        # Ask user if continue processing unprocessed files or overwrite      
        if processed_files and len(processed_files)<len(img_files):
            continue_ans = str(
                input(f"Would you like to pickup where you left off? [y/n]"))
            if "y" in continue_ans.lower():
                final_imgs = []
                final_annos = []
                for im_path, anno_path in zip(img_files, anno_files):
                    im_name_root = str(Path(im_path).name.split(Path(im_path).suffix)[0])
                    if im_name_root not in processed_files:
                        final_imgs.append(im_path)
                        final_annos.append(anno_path)
                img_files = final_imgs
                anno_files = final_annos
        elif len(processed_files) == len(img_files):
            overwrite_processed_ans = str(
                input("Reprocess all images? [y/n]")
            )
            if "n" in overwrite_processed_ans.lower():
                exit()
        print(f"\n{len(img_files)} images with {len(anno_files)} annotations from {dset_name} will be processed\n")
       
        # Multiprocessing
        with mp.Pool(processes=num_workers) as pool:
            if visualize:
                func = partial(main, img_out_dir, visualize=True)
                pool.starmap(func, zip(img_files, anno_files))
            else:
                func = partial(main, img_out_dir)
                pool.starmap(func, zip(img_files, anno_files))
                #for img, anno in zip(img_files, anno_files):
                #    main(img, anno, max_dim)
               
    print("\nAll done!")


        
        

                    
