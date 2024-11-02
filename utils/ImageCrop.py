# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 19:19:36 2023

@author: adlm
"""
from PIL import Image
import os
import time
import numpy as np
from math import sin, cos, radians
import cv2
from shapely.geometry import Polygon


class photomontage:
    def __init__(self, cfg):
        self.cfg = cfg
        self.Rotmat = {}
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        if cfg is not None:
            self.Read_Data(cfg)
            self.NewImg = Image.new(
                'RGB',
                (self.cfg['Image_size'], self.cfg['Image_size']),
                (110, 128, 142))
            
            for i in range(5):
                theta = radians(self.cfg['ZoneData'][i]['Theta'])
                c, s = cos(theta), sin(theta)
                self.Rotmat[i] = np.array([[c, s], [-s, c]])
                
        self.images = {}
        self.output_image = Image.new('RGB', (1280, 1280), (110, 128, 142))
        self.NewImg2= np.full((self.cfg['Image_size'], self.cfg['Image_size'], 3), (142, 128, 110), dtype=np.uint8)
        self.output_image2= np.full((1280, 1280, 3), (0, 0, 0), dtype=np.uint8)
        self.top_left_x = int((640 - self.cfg['ImgCX']) * np.heaviside(640 - self.cfg['ImgCX'], 0))
        self.bottom_right_x = int(1280 + (640 - self.cfg['ImgCX']) * np.heaviside(-640 + self.cfg['ImgCX'], 0))
        self.img_TLX = int(np.heaviside(-640 + self.cfg['ImgCX'], 0) * (-640 + self.cfg['ImgCX']))
        self.img_BRX = int(1280 + (-640 + self.cfg['ImgCX']) * np.heaviside(640 - self.cfg['ImgCX'], 0))
        self.top_left_y = 640 - self.cfg['ImgCY']
        self.bottom_right_y = 960 +   640 - self.cfg['ImgCY']
        self.crop = {}
        self.M = {}
        self.Resize = {}
        self.paste = {}
        for i in range(5):
            # Define the rotation matrix
            TL = self.cfg['ZoneData'][i]['Coordinates'][0]
            BR = self.cfg['ZoneData'][i]['Coordinates'][1]
            self.M[i] = cv2.getRotationMatrix2D((640, 640), self.cfg['ZoneData'][i]['Theta'], 1.0)
            self.crop[i] = (*TL, *BR)
            self.Resize[i] = (self.cfg['Resize'][i]*(BR[0] - TL[0]), self.cfg['Resize'][i]*(BR[1] - TL[1]))
            self.paste[i] = [*self.cfg['ZoneData'][i]['Location'][0:2],
                             self.cfg['ZoneData'][i]['Location'][0] + self.cfg['ZoneData'][i]['Location'][2],
                             self.cfg['ZoneData'][i]['Location'][1] + self.cfg['ZoneData'][i]['Location'][3]]

    # Function to read the provided information from cfg and Lane_Data files
    def Read_Data(self, cfg_file): 
        try:
            with open(cfg_file, 'r') as f:
                lines = f.readlines()
        except:
            print("cfg_file was not found")
            exit()
        
        Parameters = dict()
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            key, val = line.split('=')
            Parameters[key.strip()] = val.strip()
            
        def Str2int(Coord):
            for i in range(len(Coord)) :
                Coord[i] = int(Coord[i])
            return(Coord)
        
        Cfg_Data = {}
        Cfg_Data["ImgCX"] = int(Parameters["FisheyeImageCenterX"]) 
        Cfg_Data["ImgCY"] = int(Parameters["FisheyeImageCenterY"]) 
        Cfg_Data['ImgC'] = np.array([640, 640]) 
        Cfg_Data['OIC'] = np.array([Cfg_Data['ImgCX'] , Cfg_Data['ImgCY']])
        Cfg_Data["Image_size"] = int(Parameters["Image_size"]) 
        Cfg_Data["Resize"] = Str2int(Parameters["Resize"].split())        
        ZoneData = [0]*5
        for i in range(5) :
            Zdata = dict()
            ZD = Str2int(Parameters["Zone"+ str(i) + "Coordinates"].split())
            Zdata["Coordinates"] = np.reshape(ZD, (int(len(ZD)/2), 2)).tolist()
            Zdata["Theta"] = int(Parameters["Zone"+ str(i) + "Theta"])
            Zdata["Location"] = Str2int(Parameters["Zone"+ str(i) + "location"].split())
            ZoneData[i] = Zdata
        Cfg_Data["ZoneData"] = ZoneData    
        self.cfg = Cfg_Data
    
    def GenImg(self, Img):
        t1 = time.time()
        self.output_image2[self.top_left_y:self.bottom_right_y, self.top_left_x:self.bottom_right_x] = Img[:, self.img_TLX:self.img_BRX]
        t2 = time.time()
        print(f"center matching = {1000*(t2 - t1):.1f} ms")
        for i in range(5):
            img = cv2.warpAffine(self.output_image2, self.M[i], (1280, 1280), flags=cv2.INTER_NEAREST)
            self.images[i] = img[self.crop[i][1]:self.crop[i][3], self.crop[i][0]:self.crop[i][2]]
        t3 = time.time()
        print(f"rotate crop = {1000*(t3 - t2):.1f} ms")
        for i in range(5):
            if self.cfg['Resize'][i] > 1:
                self.images[i] = cv2.resize(self.images[i], self.Resize[i], interpolation=cv2.INTER_LINEAR)
            self.NewImg2[self.paste[i][1]:self.paste[i][3], self.paste[i][0]:self.paste[i][2]] = self.images[i] 
        #cv2.imwrite(os.getcwd() + '\img.jpg', self.NewImg2)
        t4 = time.time()
        print(f"resize construct = {1000*(t4 - t3):.1f} ms")
        return  self.NewImg2
    
    def GenImg2(self, Img):
        t1 = time.time()
        self.output_image2[self.top_left_y:self.bottom_right_y, self.top_left_x:self.bottom_right_x] = Img[:, self.img_TLX:self.img_BRX]
        t2 = time.time()
        print(f"center matching = {1000*(t2 - t1):.1f} ms")
        for i in range(5):
            img = cv2.warpAffine(self.output_image2, self.M[i], (1280, 1280), flags=cv2.INTER_NEAREST)
            self.images[i] = img[self.crop[i][1]:self.crop[i][3], self.crop[i][0]:self.crop[i][2]]
        t3 = time.time()
        print(f"rotate crop = {1000*(t3 - t2):.1f} ms")
        for i in range(5):
            if self.cfg['Resize'][i] > 1:
                self.images[i] = cv2.resize(self.images[i], self.Resize[i], interpolation=cv2.INTER_LINEAR)
            self.NewImg2[self.paste[i][1]:self.paste[i][3], self.paste[i][0]:self.paste[i][2]] = self.images[i] 
        #cv2.imwrite(os.getcwd() + '\img.jpg', self.NewImg2)
        t4 = time.time()
        print(f"resize construct = {1000*(t4 - t3):.1f} ms")
        return  self.NewImg2


    def Transfer(self, BBox):
        
        def point_in_rectangle(point, rectangle):
            x, y = point
            x1, y1, x2, y2 = rectangle
            if x1 <= x <= x1+x2 and y1 <= y <= y1+y2:
                return True
            else:
                return False
            
        def BBoxLocating(BBC):
            loc = 5
            for i in range(5):
                if point_in_rectangle(BBC, self.cfg['ZoneData'][i]['Location']):
                    loc = i
            return loc
        
        def TakeCoordtoRotatedImg(BBox, i, BBC):
            TargetCorner = np.array(self.cfg['ZoneData'][i]['Coordinates'][0])
            InitialCorner =  np.array(self.cfg['ZoneData'][i]['Location'][0:2])
            BBox[0:2] = (BBC - InitialCorner)/self.cfg['Resize'][i] + TargetCorner
            BBox[2:]  = BBox[2:] / self.cfg['Resize'][i]
            return BBox
        
        def RotateBBox(BBox, i):
            def get_corner_points(bbox):
                """
                Returns the four corner points of a bounding box.
                """
                x, y, w, h = bbox
                half_w, half_h = w / 2, h / 2
                top_left = [x - half_w, y - half_h]
                top_right = [x + half_w, y - half_h]
                bottom_right = [x + half_w, y + half_h]
                bottom_left = [x - half_w, y + half_h]
                return np.array([top_left, top_right, bottom_right, bottom_left])
         
            Corners = get_corner_points(BBox)
            BBX = Corners - self.cfg['ImgC']
            BBX = (np.matmul(BBX, self.Rotmat[i]) + self.cfg['OIC']).astype('int32')
            BBX = np.ravel(BBX).tolist()
            BBX.append(i)
            return BBX
        
        BBC = BBox[0:2]
        Zone = BBoxLocating(BBC)
        if Zone < 5:
            TBBox = TakeCoordtoRotatedImg(BBox, Zone, BBC)
            #print("TBBox: ", TBBox)
            return RotateBBox(TBBox, Zone)
        else:
            return None 
        
    def draw_bounding_box(self, points, image, label=None, color=None, line_thickness=3): 
        # Draw the bounding box on the image
        bbox = np.array(points).reshape(4,2).astype(int)
        cv2.polylines(image, [bbox], True, color, line_thickness)
        # Write the label on the bounding box
        # label_size, _ = cv2.getTextSize(label, font, 0.5, 2)
        # label_pos = (points[0][0], points[0][1] - label_size[1] - 5)
        # cv2.putText(image, label, label_pos, self.font, 0.5, (0, 255, 0), 2)
    
        return image
    
    # plot_one_box(OBB, im1s, label=label, color=colors[int(cls)], line_thickness=1)
    
    # plot_one_box(x, img, color=None, label=None, line_thickness=3):
        
    # img = Image.fromarray(img)
    # draw = ImageDraw.Draw(img)
    # line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    # draw.rectangle(box, width=line_thickness, outline=tuple(color))  # plot
    # if label:
    #     fontsize = max(round(max(img.size) / 40), 12)
    #     font = ImageFont.truetype("Arial.ttf", fontsize)
    #     txt_width, txt_height = font.getsize(label)
    #     draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=tuple(color))
    #     draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    # return np.asarray(img)



    
    def NanMaxSup(self, detections, threshold):

        def compute_overlap(box1, box2):
            """
            Compute the overlap between two oriented bounding boxes.
            Each box is represented by a list of eight numbers, which are the x and y coordinates
            of the four corners of the box in clockwise order starting from the top-left corner.
            """
            
            # Convert the boxes to polygons
            poly1 = np.array(box1).reshape(4, 2)
            poly2 = np.array(box2).reshape(4, 2)
            
            # Compute the intersection of the polygons
            intersection = Polygon(poly1).intersection(Polygon(poly2)).area
            
            # Compute the union of the polygons
            union = Polygon(poly1).union(Polygon(poly2)).area
            
            # Compute the overlap as the intersection over the union
            overlap = intersection / union
            
            return overlap
        
        """
        Perform non-maximum suppression on the input detections list based on the given threshold.
        Each detection in the input list is a list with the following format:
        [x1, y1, x2, y2, x3, y3, x4, y4, score, class_label]
        where (x1, y1), (x2, y2), (x3, y3), and (x4, y4) are the four corners of the oriented bounding box,
        score is the class probability, and class_label is the object class label.
        """
        # Sort the detections by decreasing score
        detections = sorted(detections, key=lambda d: d[9], reverse=True)
        
        # Initialize list of selected detections
        selected_detections = []
        
        while detections:
            # Select the detection with the highest score and remove it from the list
            best_detection = detections.pop(0)
            selected_detections.append(best_detection)
            
            # Iterate over the remaining detections
            remaining_detections = []
            for detection in detections:
                if (detection[8] != best_detection[8]):
                #and (detection[8] == 0 or best_detection[8] == 0):
                    # Compute the overlap between the two bounding boxes
                    overlap = compute_overlap(best_detection[:8], detection[:8])
                    
                    # If the overlap exceeds the threshold, remove the detection from the list
                    if overlap < threshold:
                        remaining_detections.append(detection)
                else:
                    remaining_detections.append(detection)
            detections = remaining_detections
            
        return selected_detections
        
  





