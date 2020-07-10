#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jager
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as albu

from torch.utils.data import TensorDataset, DataLoader,Dataset
from albumentations.pytorch import ToTensor
from catalyst.dl.runner import SupervisedRunner


from catalyst.dl import utils

def get_img(x, path):
    
    data_folder = f"{path}"
    image_path = os.path.join(data_folder, x)
    if '.tif' in x:
        img = cv2.imread(image_path)
    else:
        img = utils.imread(image_path)
    return img

# Thanks to Andrew Lukyanenko for some useful methods. Annotation saved
    ###
def rle_decode(mask_rle: str = '', shape: tuple = (1120, 1120)):
    '''
    Decode rle encoded mask.
    
    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def make_mask(df: pd.DataFrame, image_name: str='img.jpg', shape: tuple = (1120, 1120)):
 
    encoded_masks = df.loc[df['ImageID'] == image_name, 'EncodedPixel']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask
            
    return masks


def to_tensor(x, **kwargs):
   
    return x.transpose(2, 0, 1).astype('float32')


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def post_process(probability, threshold, min_size):
 
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1120, 1120), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


    ###
    
def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 14
    class_dict = {0: 'Buildings', 1: 'Car', 2: 'Trees', 3: 'Water'}
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 5, figsize=(34, 54))

        ax[0].imshow(image)
        for i in range(4):
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].set_title(f'Mask {class_dict[i]}', fontsize=fontsize)
    else:
        f, ax = plt.subplots(2, 5, figsize=(24, 12))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
                
        for i in range(4):
            ax[0, i + 1].imshow(original_image)
            ax[0, i + 1].imshow(original_mask[:, :, i], alpha=0.7, cmap='gray')
            ax[0, i + 1].set_title(f'Original mask {class_dict[i]}', fontsize=fontsize)
        
        ax[1, 0].imshow(image)
        ax[1, 0].set_title('Transformed image', fontsize=fontsize)
       
        for i in range(4):
            ax[1, i + 1].imshow(image)
            ax[1, i + 1].imshow(mask[:, :, i], alpha=0.7, cmap='gray')
            ax[1, i + 1].set_title(f'Transformed mask {class_dict[i]}', fontsize=fontsize)
            
            
def visualize_with_raw(image, mask, original_image=None, original_mask=None, raw_image=None, raw_mask=None):
    fontsize = 14
    class_dict = {0: 'Buildings', 1: 'Car', 2: 'Trees', 3: 'Water'}

    f, ax = plt.subplots(3, 5, figsize=(44, 32))

    ax[0, 0].imshow(original_image)
    ax[0, 0].set_title('Original image', fontsize=fontsize)

    for i in range(4):
        ax[0, i + 1].imshow(original_mask[:, :, i])
        ax[0, i + 1].set_title(f'Original mask {class_dict[i]}', fontsize=fontsize)

    ax[1, 0].imshow(raw_image)
    ax[1, 0].set_title('Original image', fontsize=fontsize)

    for i in range(4):
        ax[1, i + 1].imshow(raw_mask[:, :, i])
        ax[1, i + 1].set_title(f'Raw predicted mask {class_dict[i]}', fontsize=fontsize)
        
    ax[2, 0].imshow(image)
    ax[2, 0].set_title('Transformed image', fontsize=fontsize)

    for i in range(4):
        ax[2, i + 1].imshow(mask[:, :, i])
        ax[2, i + 1].set_title(f'Predicted mask with processing {class_dict[i]}', fontsize=fontsize)
            
            
def plot_with_augmentation(image, mask, augment):
    
    #Plot augmented image with augmentation using Albumentations
    
    augmented = augment(image=image, mask=mask)
    deformed_image = augmented['image']
    deformed_mask = augmented['mask']
    visualize(deformed_image, deformed_mask, original_image=image, original_mask=mask)
    
    
sigmoid = lambda x: 1 / (1 + np.exp(-x))

class CityDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None, path: str = '',
                 transforms = albu.Compose([albu.HorizontalFlip(),ToTensor()]),
                preprocessing=None):
        self.df = df
        self.path = path
        if datatype != 'test':
            self.data_folder = f"{self.path}/image_train"
        else:
            self.data_folder = f"{self.path}/test"
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)
