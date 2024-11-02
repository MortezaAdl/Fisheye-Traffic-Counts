# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:58:19 2023

@author: adlm
"""

import cv2
import os
from Counter.Intersection import FishEye
from matplotlib import pyplot as plt
from pickle import load
import numpy as np

class TrajPlotter:
    def __init__(self, color):
        self.trajfile = os.getcwd() + '/Counter/Training/TempTrajList.pkl'
        with open(self.trajfile, 'rb') as f:
            self.TempTrajList = load(f)
        self.color = color
        
    def plot_curve_through_points(self, points, image):
        # Create an array of points to draw the curve
        curve_points = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [curve_points], isClosed=False, color=self.color, thickness=2)
        
        return self.image
    
    def plot_trajectories(self, image):
        for tracklist in self.TempTrajList:
            for tracks in tracklist:
                for track in tracks:
                    if track:
                        TempTraj = FishEye(np.array(track)).astype("int")
                        image = self.plot_curve_through_points(TempTraj, image)
        
        return image

              
                
