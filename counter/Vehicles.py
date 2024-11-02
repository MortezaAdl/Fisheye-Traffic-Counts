# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:12:40 2022

@author: Morteza_Adl
"""
import os
import numpy as np
import cv2
from Counter.Tracks import Tracker
from Counter.DataCollection import CollectData
from Counter.Intersection import ZoneData
from Counter.Clustering import TrajCluster, PTC


class Surveillance:
    def __init__(self, FilteringDistance, TrainingMode, TPL, Image_Folder):
        self.Vehicles        = Tracker(FilteringDistance, Image_Folder)
        self.TPL = TPL
        self.NCV = 0
        self.NCB = 0
        self.NNB = 0
        self.traffic = []
        self.Counts = np.zeros((4,4), dtype=int)
        self.Car = np.zeros((4,4), dtype=int)
        self.Bus = np.zeros((4,4), dtype=int)
        self.Truck = np.zeros((4,4), dtype=int)
        self.TrainingMode    = TrainingMode
        self.TrajPrediction  = False
        cwd = os.getcwd()

        if TrainingMode:
            print("********** Training Mode is on **********")
            if not os.path.exists(cwd + '/Counter/Training'):
                os.mkdir(cwd + '/Counter/Training')
            self.CollectData = CollectData(ZoneData, self.TPL)
            
        elif os.path.exists(cwd + '/Counter/Training/TempTrajList.pkl'):
                self.PBTC = PTC()
                self.TrajPrediction  = True
    
    def Learn_Patters(self):
        for track in self.Vehicles.tracks:
            if track.Type == "Classifiable" and not hasattr(track, 'Saved'):
                track.Saved = True
                self.CollectData.AddTrack(track)
                
        if self.CollectData.ReadyForTraining :
            print("\n************ learning patterns is started ***********\n")
            TrajCluster(self.CollectData.Tracklist)
            self.TrainingMode = False
            self.TrajPrediction = True
            self.PBTC = PTC()
    
    def _Predict_Broken_trajectory_class(self):
        for track in self.Vehicles.tracks:
            if track.Type == "Broken":
                self.PBTC.PredictTrajClass(track)
    
    
    def _Count_Vehicles(self):
        def count(track):
            self.Counts[track.Entrance_Zone - 1][track.Exit_Zone - 1] += 1
            self.traffic.append((track.Entrance_Zone, track.Exit_Zone))
            if track.cls == 2:
                self.Car[track.Entrance_Zone - 1][track.Exit_Zone - 1] += 1
            elif track.cls == 4:
                self.Bus[track.Entrance_Zone - 1][track.Exit_Zone - 1] += 1
            elif track.cls == 5:
                self.Truck[track.Entrance_Zone - 1][track.Exit_Zone - 1] += 1
                
            track.Counted = True
            
        for track in self.Vehicles.tracks:
            if track.Counted == False:
                if track.Type =="Classifiable":
                    self.NCV += 1
                    count(track)
                    
                elif track.Type == "Classifiable_Broken":
                    self.NCB += 1
                    count(track)
                
                elif track.Type == "UnClassifiable_Broken":
                    self.NNB += 1
                    track.Counted = True
                    
                    
    def Update(self, FrameData):
        self.traffic = []
        self.Vehicles.Match(FrameData)
        
        if self.TrainingMode:
           self.Learn_Patters()
           
        if self.TrajPrediction:
            self._Predict_Broken_trajectory_class()
            
        self._Count_Vehicles()            
        self.Vehicles.CleanTracks()
  
               
       
    

 
    