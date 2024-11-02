# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 10:15:25 2022

@author: Morteza_Adl
"""
from Counter.Intersection import *
import cv2


class Track:

    def __init__(self, ID, AppearanceFrame, BBoxCent, BBox, cls, Filtering_Distance, Image_folder):
        self.ID = ID
        self.BBoxCenters = [BBoxCent]
        self.BBoxes = [BBox]
        self.Frames = [AppearanceFrame]
        self.Locations = [Distance_Filter(DeFishEye(BBoxCent)[0], Filtering_Distance)]
        self.Zones = [Find_Zone(self.Locations[0])]
        self.classes   = [cls]
        self.cls = cls
        self.Filtering_Distance = Filtering_Distance
        self.time_since_update = 0
        self.state = "Tentative"  # Tentative, Confirmed
        self.save = 0
        self.Entrance_Zone = "Unknown"
        self.Exit_Zone = "Unknown"
        self.Movement_dir = "Arriving"
        self.Distances = [Distance(self.Locations[0], Rec_Intersection_center)]
        self.Counted = False
        self.Type = "Unknown"   # long, broken, short
        self.Traj = {0: "U-Turn", 1: "Right-Turn", 2: "Straight", 3: "Left-Turn"}
        self.Trajectory = "Unknown"
        self.Image_folder = Image_folder

        
      
    def _UpdateLocation(self):
        self.Locations.append(Distance_Filter(DeFishEye(self.BBoxCenters[-1])[0], self.Filtering_Distance))

    def _UpdateZone(self):
        self.Zones.append(Find_Zone(self.Locations[-1]))
        
    def _UpdateDistance(self):
        self.Distances.append(Distance(self.Locations[-1], Rec_Intersection_center))
        
    def _UpdateMoveDir(self):
        
        if self.Distances[-1] < self.Distances[-2] and not(0 in self.Zones) and self.Movement_dir == "Unknown":
            self.Movement_dir = "Arriving"
        
        if self.Zones[-1] == 0:
            if self.Movement_dir == "Arriving":
                self.Entrance_Location = self.Locations[-2]
            self.Movement_dir = "Inside Intersection"
           
        
        if 5 > self.Zones[-1] > 0 and self.Movement_dir == "Inside Intersection":
            if not(self.Entrance_Zone == "Unknown"):
                if Distance(self.Locations[-1], self.Entrance_Location) > 5:
                    self.Movement_dir = "Leaving"
            else:
                self.Movement_dir = "Leaving"
                

                    

    def ClassifyTrajectory(self):
        self.Trajectory = self.Traj[(self.Exit_Zone - self.Entrance_Zone) % 4]
            


    def Update(self, detection):
        """update the Vehicle's states.
        Parameters
        ----------
        detection : Detection
            The associated detection.
        """
        self.BBoxes.append(detection[2:6])
        self.BBoxCenters.append(detection[2:4])
        self.Frames.append(detection[0])
        self.classes.append(detection[6])
        self._UpdateLocation()
        self._UpdateZone()
        self._UpdateDistance()
        self._UpdateMoveDir()
         
        if self.Entrance_Zone == "Unknown":
            if self.Movement_dir == "Arriving" and 5 > self.Zones[-2] > 0:
                self.Entrance_Zone = self.Zones[-2]
                
        if (self.Exit_Zone == "Unknown") and 5 > self.Zones[-1] > 0:
            if self.Movement_dir == "Leaving" :
                    self.Exit_Zone = self.Zones[-1]
                
                    
    def Video(self):
        
        def xywh_to_xyxy(bbox):
            cx, cy, w, h = bbox
            xmin = int(cx - w/2)
            ymin = int(cy - h/2)
            xmax = int(cx + w/2)
            ymax = int(cy + h/2)
            return [xmin, ymin, xmax, ymax]   
        
        names = {2:'Car', 5:'Bus', 7:'Truck', 4:'Van', "Unknown":'Unknown'}
        images = [img for img in os.listdir(self.Image_folder) if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(self.Image_folder, images[0]))
        images = images[int(self.Frames[0]): int(self.Frames[-1]) + 1]
        height, width, layers = frame.shape
        label = names[self.cls] + str(self.ID) 
        video_name = cwd + '/Videos/' + label + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))
        j = 0
        for x in range(len(images)):
            image = cv2.imread(os.path.join(self.Image_folder, images[x]))
            if self.Frames[j] - self.Frames[0] == x:
                if j < len(self.Frames)-1:
                    j += 1
                   
                x1, y1, x2, y2 = [int(i) for i in xywh_to_xyxy(self.BBoxes[j])]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=0)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(image, (x1, y1), c2, (0, 0, 0), -1, cv2.LINE_AA)  # filled
                cv2.putText(image, label, (x1, y1 - 2), 0, 1, [225, 255, 255], thickness=0, lineType=cv2.LINE_AA)
                text_0 = "Trajectory: " + str(self.Trajectory)
                text_1 = "Track type:" + str(self.Type)
                cv2.putText(image, text_0, (5, 30), 0, 0.7, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(image, text_1, (5, 60), 0, 0.7, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

            video.write(image)
        cv2.destroyAllWindows()
        video.release()
