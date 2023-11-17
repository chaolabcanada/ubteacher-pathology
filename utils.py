import glob 
import json
import os
from typing import Dict, Tuple, List, Set
import multiprocessing as mp

import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
from matplotlib import patches
from detectron2.data import transforms as T


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


def vis_boxes(image: np.array, bboxes: List, box_names=None):   
    '''Return visuals for bboxes on image.
    Args:
    image - np.array with shape (h, w, c),
    bboxes - list of [x1, y1, x2, y2] coordinates,
    box_names - optional list of names for each box.
    '''
    fig, ax = plt.subplots()
    N = len(bboxes) * 5
    cmap = plt.cm.get_cmap('hsv', N)
    # Pad image with border
    padded = np.pad(image, pad_width=((5,5), (5,5), (0,0)), mode='constant', constant_values=0)
    ax.imshow(padded)
    # Draw bbox over all kept contours
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
                    (x1, y1), width, height, linewidth=1, edgecolor=cmap(i), facecolor='none'
                )
        ax.add_patch(rect)
        if box_names:
            ax.annotate(box_names[i], (x1, y1), color=cmap(i), ha="left", va="top")
    plt.show(); plt.close()
    
