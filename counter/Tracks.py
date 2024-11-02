# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:15:25 2022

@author: Morteza_Adl
"""
import sys
from counter.Track import Track
import numpy as np 

class Tracker:
    def __init__(self, Filtering_Distance, Image_Folder):
        self.Filtering_Distance = Filtering_Distance
        self.tracks             = []
        self.Image_Folder = Image_Folder
        
    def Initiate_Track(self, det):   
        self.tracks.append(Track(det[1], det[0], det[2:4], det[2:6], det[6], self.Filtering_Distance, self.Image_Folder))
       
    def Match(self, detections):
        
        """Perform Maching and track management.
        Parameters
        ----------
        detections : A list of detections in a frame at the current time step.
        """
        # Associate tracks using IDs.
        detections_List = detections.copy() 
        for track in self.tracks:
            track.time_since_update += 1
            for i in range(len(detections_List)):
                if (track.ID == detections_List[i][1]):
                    track.time_since_update = 0
                    if track.Counted == False:
                        track.Update(detections_List[i])
                    detections_List = np.delete(detections_List, i, 0) 
                    break
                
            if track.Movement_dir == "Leaving" and not(track.state == "Confirmed"):
                track.state = "Confirmed"
                if track.classes.count(4) > 7:
                    track.cls = 4
                elif track.classes.count(5) > 7:
                    track.cls = 5
                    
                if not(track.Entrance_Zone == "Unknown") and not(track.Exit_Zone == "Unknown") :
                    track.Type = "Classifiable"
                    track.ClassifyTrajectory()  
                else :
                    track.Type = "Broken"  
                                      
        for det in detections_List:
            self.Initiate_Track(det)
        
    def CleanTracks(self):
        self.tracks = [track for track in self.tracks if track.time_since_update < 20]  
            
     

    
     
                        
                
                
        
   
        
