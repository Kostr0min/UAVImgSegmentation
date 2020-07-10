#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kostr0min
"""
import pandas as pd
import os
import numpy as np
import cv2 as cv

houses_df = pd.DataFrame(columns=['ImageID', 'House_number', '[x_min,x_max,y_min,y_max]'])

class Create_houses_df:
    def __init__(self, img, color):
        self.image = img
        self.color = color
    
    def get_edge(self, image):
        color = self.color
        lower = np.array(color)
        upper = np.array(color)

        img = np.copy(image)
        thresh = cv.inRange(img, lower, upper)
        th = np.copy(thresh)
        contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        return(contours)
    
    def create_coordinate_df(self, df=houses_df):
        image = self.image
        img_cont = self.get_edge(image)
        for i,cont in enumerate(img_cont):
            img_cont[i] = img_cont[i].reshape((-1, 2)).astype(np.int32)
            x_min = img_cont[i][:, 0].min()
            x_max = img_cont[i][:, 0].max()
            y_min = img_cont[i][:, 1].min()
            y_max = img_cont[i][:, 1].max()
            df = df.append({'ImageID': 0, 'House_number':i, '[x_min,x_max,y_min,y_max]':  [x_min,x_max,y_min,y_max]}, ignore_index= True)
        return(df)

