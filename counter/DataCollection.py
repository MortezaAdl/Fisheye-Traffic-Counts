# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:24:35 2023

@author: adlm
"""
from numpy import zeros, heaviside, all
from numpy import sum as npsum
from pickle import load, dump
from os import getcwd
from os.path import exists



class CollectData:
    def __init__(self, ZoneData, MinTrackNum):
        self.OVN = 0 # observed vehicle number
        self.Train_datanum =  self._Define_Training_datanum(ZoneData, MinTrackNum)   
        self._loadPreviousData()
        self.progress = 0
       
               
    def _Define_Training_datanum(self, ZoneData, MinTrackNum):
        Train_datanum = zeros((4, 4))
        for i in range(4):
            if ZoneData[i]['Status'] == 'Active' and int(ZoneData[i]['ASLN']) > 0:
                for j in range(4):
                    if ZoneData[j]['Status'] == 'Active' and int(ZoneData[j]['LSLN']) > 0:
                        if (i - j) % 4 == 1:
                            Train_datanum[i][j] = MinTrackNum
                            
                        elif (i - j) % 4 == 2:
                            Train_datanum[i][j] = MinTrackNum * max(int(ZoneData[i]['ASLN']) - heaviside(int(ZoneData[i]['ASLN']) - 2, 0), 1)
                            
                        elif (i - j) % 4 == 3: 
                            Train_datanum[i][j] = MinTrackNum  

        if all(Train_datanum == 0):
            raise ValueError("Please enter number of the lanes in the Counter/cfg.txt file to start data collection.")
        return Train_datanum
    
    def _loadPreviousData(self):
        # Specify the file path
        self.file_path = getcwd() + 'Counter/Training/Tracklist.pickle'
        # Check if the file exists
        if exists(self.file_path):
            # Load the list of objects from the file
            with open(self.file_path, 'rb') as file:
                self.Tracklist = load(file)
                self.Tracklist_num = [[len(self.Tracklist[i][j]) for j in range(4)] for i in range(4)]
                print("Previous data size: ", self.Tracklist_num)
            self._Check_Data_Adequacy() 
            
        else:
            self.Tracklist = [[] for _ in range(4)]  # Generate an empty list for each row
            for row in self.Tracklist:
                row.extend([[] for _ in range(4)])  # Extend each row with empty lists for each column
            self.Tracklist_num = zeros((4, 4))
            self.ReadyForTraining = False 
    
    def _Check_Data_Adequacy(self):
        if npsum(self.Train_datanum - self.Tracklist_num) <= 0 or self.OVN > 10*npsum(self.Train_datanum): 
            self.ReadyForTraining = True
        else:
            self.ReadyForTraining = False
        
    def AddTrack(self, track):
        self.OVN += 1
        i = track.Entrance_Zone - 1
        j = track.Exit_Zone - 1
        if len(self.Tracklist[i][j]) < self.Train_datanum[i][j]:
            self.Tracklist[i][j].append(track)
            self.Tracklist_num[i][j] += 1
            self._Check_Data_Adequacy() 
        # print(self.Tracklist_num)
        # print(self.Train_datanum)
        # print(self.OVN)
        self.progress = int(100*max(npsum(self.Tracklist_num) / npsum(self.Train_datanum),self.OVN/10/npsum(self.Train_datanum)))
        
    def SaveData(self):
        # Save the list of objects to a file
        with open(self.file_path, 'wb') as file:
            print("Training data matrix: ", self.Tracklist_num, "\n")
            dump(self.Tracklist, file)
    




