# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:24:26 2023

@author: adlm
"""

from Counter.Intersection import ImgCX, ImageWidth, ImageHeight, FishEye
import numpy as np
import sys
from colorama import Fore
import cv2
import os
from matplotlib import pyplot as plt
from pickle import load


OIS_size = [ImageWidth, ImageHeight]
new_img_size = min(OIS_size)
img_center = ImgCX

 
def calculate_angle(V):
    vector = V.copy()
    center = np.array([new_img_size/2, -new_img_size/2])
    vector[1] = - vector[1] 
    vector1 = vector - center
    vector2 = np.array([0, new_img_size/2])
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(vector1, vector2)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def rotate_coordinate(C, angle_deg):
    coord = C.copy()
    coord[1] = - coord[1]
    center = np.array([new_img_size/2, -new_img_size/2])
    angle_rad = np.radians(angle_deg)
    coord = coord - center
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                 [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_coord = np.dot(rotation_matrix, coord)
    rotated_coord = rotated_coord + center
    rotated_coord[0] =  round(rotated_coord[0])
    rotated_coord[1] = - round(rotated_coord[1])
    return rotated_coord


def NMS(bboxes, threshold):
    
    def calculate_iou(box1, box2, threshold):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
    
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
         
        iou = intersection / (area1 + area2 - intersection)
        if iou < threshold and intersection/min(area1, area2) > 0.9:
            iou = threshold + 0.1
            
        return iou

    # Sort bounding boxes by confidence in descending order
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)

    # Initialize list of selected bounding boxes
    selected_boxes = []

    # Iterate through all bounding boxes
    while len(bboxes) > 0:
        # Select the bounding box with highest confidence
        box = bboxes.pop(0)

        # Add the box to the list of selected boxes
        selected_boxes.append(box)

        # Compute the IoU (Intersection over Union) between the selected box and all other boxes
        ious = [calculate_iou(box, b, threshold) for b in bboxes]

        # Remove boxes that overlap significantly with the selected box
        for i in range(len(bboxes)):
            if ious[i] > threshold:
                bboxes[i] = None

        bboxes = [b for b in bboxes if b is not None]

    return selected_boxes

    
def PrintCounts(VD):
    total = 0
    line = ""
    Strline = ""
    for i in range(1, 5):
        for j in range(1, 5):
            value = VD.Counts[i-1][j-1]
            letter = f"{Fore.WHITE}{i}{j}:{Fore.YELLOW}{value}\t" # Corrected the line variable
            let = str(i) + str(j) + ":" + str(value) + " "
            line += letter
            Strline += let
            total += value
    
    line += f"Total:{Fore.CYAN}{total}"
    if VD.TrainingMode == True:
        line += f"  {Fore.WHITE}Data collection progress for training:{Fore.RED}{VD.CollectData.progress} %"
    sys.stdout.write("\r")        
    sys.stdout.write(line) 
    sys.stdout.flush()
    return Strline


class TrajPlotter:
    def __init__(self, color):
        self.trajfile = os.getcwd() + '/Counter/Training/TempTrajList.pkl'
        with open(self.trajfile, 'rb') as f:
            self.TempTrajList = load(f)
        self.color = color
        
    def plot_curve_through_points(self, points, image, color):
        # Create an array of points to draw the curve
        curve_points = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [curve_points], isClosed=False, color=color, thickness=2)
        
        return image
    
    def plot_trajectories(self, image, traffic):
        for tracklist in self.TempTrajList:
            for tracks in tracklist:
                for track in tracks:
                    if track:
                        TempTraj = FishEye(np.array(track)).astype("int")
                        image = self.plot_curve_through_points(TempTraj, image, self.color[0])

        for i in range(len(traffic)):
            for track in self.TempTrajList[traffic[i][0] - 1][traffic[i][1] - 1]:
                TempTraj = FishEye(np.array(track)).astype("int")
                image = self.plot_curve_through_points(TempTraj, image, self.color[1])

        return image
    

class EmissionEstimator:
    def __init__(self, emisionlevel):
        self.emisionlevel = emisionlevel
        self.max_emission = 10000
        self.Emission_list = [0, 0, 0, 0]

    def Update_Estimate(self, car, bus, truck):
        self.Emission_list = np.zeros(4)
        for zone in range(4):
            vehicle_num = np.array([np.sum(car[:, zone]) + np.sum(car[zone, :]), 
                                    np.sum(bus[:, zone]) + np.sum(bus[zone, :]), 
                                    np.sum(truck[:, zone]) + np.sum(truck[zone, :])])
            self.Emission_list[zone] = np.dot(vehicle_num, self.emisionlevel) 
      
    def add_bar_plot(self, original_image):

        # Create a blank canvas for the bar plot
        bar_plot = np.zeros((original_image.shape[0], 400, 3), dtype=np.uint8)

        # Normalize emission values for better visualization
        normalized_emissions = [e / self.max_emission for e in self.Emission_list]

        # Calculate the width of each bar
        bar_width_factor = 0.2  # Adjust this factor to control the width of the bars
        bar_width = int(bar_plot.shape[1] * bar_width_factor)

        # Define color thresholds
        thresholds = [0.5, 0.8]

        # Clip normalized_emissions values to be within [0, 1]
        normalized_emissions = np.clip(normalized_emissions, 0, 1)

        # Assign colors based on thresholds
        bar_colors = np.empty(4, dtype=object)
        for i in range(4):
            if normalized_emissions[i] > thresholds[1]:
                bar_colors[i] = (0, 0, 255)  # Blue
            elif normalized_emissions[i] > thresholds[0]:
                bar_colors[i] = (0, 255, 255)  # Cyan
            else:
                bar_colors[i] = (0, 255, 0)  # Green
        
        # Draw vertical bars on the bar plot
        pad = 20
        for i, (width, color) in enumerate(zip(normalized_emissions, bar_colors)):
            cv2.rectangle(bar_plot, (pad+i * (bar_width + 10), (bar_plot.shape[0]-100)),
                        (pad+(i + 1) * (bar_width + 10) - 10, int((bar_plot.shape[0]-100) - width * (bar_plot.shape[0]-100))), color, thickness=cv2.FILLED)
            cv2.putText(bar_plot, f"{int(self.Emission_list[i])}", (pad+i * (bar_width + 10), bar_plot.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bar_plot, f"Zone {i+1}", (pad+i * (bar_width + 10), bar_plot.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar_plot, "Co2 Emission level (g/km.min)", (pad, bar_plot.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    

        # Combine the bar plot with the original image
        combined_image = np.hstack((bar_plot, original_image))

        # Add horizontal text labels under each bar
        for i, street_name in enumerate(['Street 1', 'Street 2', 'Street 3', 'Street 4']):
            # Calculate the position for text
            text_position = (i * (bar_width + 10) + (i + 1) * (bar_width + 10) - 20,
                            int(bar_plot.shape[0] + original_image.shape[0] // 10))
            
            # Use cv2.putText for text
            cv2.putText(combined_image, street_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        return combined_image

    



    
