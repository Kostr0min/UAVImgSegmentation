#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kostr0min
"""

import numpy as np
import torch
from typing import List, Union
import albumentations as albu
from functools import partial

def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    classes: List[str] = None,
    eps: float = 1e-7,
    
) -> Union[float, List[float]]:
    
    if classes is not None:
        # if classes are specified we reduce across all dims except channels
        _sum = partial(torch.sum, dim=[0, 2, 3])
    else:
        _sum = torch.sum
    intersection = _sum(targets * outputs)
    union = _sum(targets) + _sum(outputs)
    iou = (intersection + eps * (union == 0)) / (union - intersection + eps)
    return iou


def augmentation(image, augment):
    augmented = augment(image=image)
    image_flipped = augmented['image']
    return(image_flipped)

# def show_house(numb,img,color = [230,   25, 75]):
#     x_min,x_max,y_min,y_max = list(houses_df[houses_df['House_number'] == numb]['[x_min,x_max,y_min,y_max]'].values)[0]
#     if (x_max - x_min)*(y_max - y_min) <= 15:
#         return(np.array[0,0])
#     house = img[y_min:y_max, x_min:x_max, :]

#     lower = np.array(color)
#     upper = np.array(color)
#     thresh = cv.inRange(house, lower, upper)
#     return(thresh)

SMOOTH = 1e-6
def iou_numpy(outputs: np.array, labels: np.array):
    
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return iou  # Or thresholded.mean()

def add_boarder(img, shape = (500,500)):
    ret = np.full((shape[0],shape[1]), 0, dtype='uint8')
    ret[int(ret.shape[0]/2 - img.shape[0]/2):int(ret.shape[0]/2 + img.shape[0]/2),int(ret.shape[1]/2 - img.shape[1]/2):int(ret.shape[1]/2 + img.shape[1]/2)] = img
    return(ret)

def cmp_iou(refer_numb, source_numb):
    refer_h = show_house(refer_numb)
    source_h = show_house(source_numb)
    new_shape = ((refer_h.shape[0] + source_h.shape[0])//2,(refer_h.shape[1] + source_h.shape[1])//2)
    refer_h = augmentation(refer_h, albu.Resize(new_shape[0],new_shape[1]))
    source_h = augmentation(source_h, albu.Resize(new_shape[0],new_shape[1]))
    jacc = iou_numpy(refer_h, source_h)
    return(jacc)

