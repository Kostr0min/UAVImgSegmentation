#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:46:16 2020

@author: jager
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import statistics as stat
import cv2 as cv
from create_df import Create_houses_df


sigmoid = lambda x: 1 / (1 + np.exp(-x))

class Select_house:
    
    def __init__(self, dataset):
        
        self.dataset = dataset
    
    def quality_index(self, x,y):
        return([(x,y),(1-x) + y])
    
    def cone(self, dataset, x, y):
        
        subset = self.dataset[np.where(self.dataset[:,0] <= x)[0]]
        subset = subset[np.where(subset[:,1] >= y)[0]]
        return(subset)   
    
    def Pareto_opt(self, dataset, _type = 'minmax'):
        dataset = self.dataset
        edge_points = []
        max_points = []
        for point in dataset:
            max_points.append(self.quality_index(point[0],point[1]))
        
        #Firstly, need to estimate: Xmin,Xmid,Xmax and Ymin,Ymid,Ymax
        
        x_min,x_max= dataset[:,0].min(),dataset[:,0].max() 
        y_min,y_max = dataset[:,1].min(),dataset[:,1].max()
        x_mid = dataset[np.where(dataset[:,1] == y_max)[0]][:,0]
        y_mid = dataset[np.where(dataset[:,0] == x_min)[0]][:,1]
        
        #And now, select all items which have a coord X less than x_mid and a coord Y much than y_mid
        
        subset = dataset[np.where(dataset[:,0] <= x_mid)[0]]
        subset = subset[np.where(subset[:,1] >= y_mid)[0]]
        
        # On this step, need to select the points that give the Pareto edge using "cone" rules
        #high_to_low_y = np.flip(np.sort(subset[:,1]))
        high_to_low = subset[subset[:,0].argsort(kind='mergesort')]
        for point in high_to_low:
            print(type(point[0]))
            sets = self.cone(subset, point[0], point[1])
            if sets.size == 2:
                edge_points.append(point)
        return(np.array(edge_points), np.array(max_points))
    