import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, xywh2xyxy, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.sort import *
from PIL import ImageFont
import csv
import os
from counter.Vehicles import Surveillance
from counter.General import *
from counter.Intersection import ZoneData
import datetime


def detect(save_img=False):
    source, weights, view_img, save_csv, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_csv, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_csv else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        
    # Get names and colors
    names =  ['', '', 'car', '', 'bus', 'truck']
    #names =  ['Bus', 'Bike', 'Car', 'Pedestrain', 'Truck']
    colors = [[255, 0, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [0, 150, 150], [255, 0, 255]]
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
        
    if view_img:
        cv2.namedWindow('Real-Time Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-Time Video', 800, 800)
    
    VD = Surveillance(50, opt.LearnPatterns, opt.TracksPerLane, source)
    print("Started at:", datetime.datetime.now().strftime("%H:%M:%S"))
    print("\nNumber of Vehicles: Zone[i][j]:Counts\n")
    
    TrajPlot = TrajPlotter([(0, 0, 0), (0, 0, 255)]) # Black, Red
    Classes = [2, 4, 5] # class types: car, bus, and truck
    frame_idx = 0
    LastCounts = np.zeros((4,4), dtype=int)
    LastCar = np.zeros((4,4), dtype=int)
    LastBus = np.zeros((4,4), dtype=int)
    LastTruck = np.zeros((4,4), dtype=int)
    Emission = EmissionEstimator([192, 307, 500])
    CountsLine = ""
    print("Start time: ", datetime.datetime.now().strftime("%H:%M:%S"))
    for path, img, im0s, vid_cap, dis in dataset:
        frame_idx += 1
        Det = []
        FD = []
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=Classes, agnostic=opt.agnostic_nms)
        

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        for i, det in enumerate(pred):
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                csv_path = str(save_dir / 'labels' / 'Detection') 
                if len(det):
                    dets_to_sort = []
                    # Rescale boxes from img_size to im0 size
                    scale_with = list(im0.shape)
                    if dis > 0:
                        scale_with[1] = scale_with[0]

                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], scale_with).round() 
              
                    if dis > 0:
                        det[:, 0] = det[:, 0] + dis  
                        det[:, 2] = det[:, 2] + dis


                    for *xyxy,conf,detclass in det.cpu().detach().numpy():
                        dets_to_sort.append([*xyxy, conf, detclass])
                         
        tracked_dets = sort_tracker.update(np.array(dets_to_sort))


        for track in tracked_dets:
            xywh = xyxy2xywh(np.array([track[0:4]]))[0] # normalized xywh
            Det.append([*xywh.round(), track[4], track[8]]) 
            FD.append([frame_idx + 1, track[8], *xywh.round(), track[4]])
            
        VD.Update(FD)
        
        if frame_idx % 1 == 0:
            CountsLine = PrintCounts(VD)
        
        # if frame_idx % 600 == 0:
        #     Emission.Update_Estimate(VD.Car - LastCar, VD.Bus - LastBus, VD.Truck - LastTruck)
        #     LastCar = VD.Car.copy()
        #     LastBus = VD.Bus.copy()
        #     LastTruck = VD.Truck.copy()
            
                            
        # draw boxes for visualization
        if len(tracked_dets) > 0 and (view_img or save_img):
            for bbox in tracked_dets:
                xyxy, cls, id = bbox[0:4], int(bbox[4]), int(bbox[8])    
                label = str(id) + ":"+ names[cls] 
                plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=1)
        
            
        if view_img:
            # pad a rectangular box on the upper side of the image
            im0s = TrajPlot.plot_trajectories(im0s, VD.traffic)
            Shown_img = im0s.copy()
            #Shown_img = Emission.add_bar_plot(Shown_img)
            for i in range(4):
                Center = np.int32(ZoneData[i]["Center"])
                cv2.putText(Shown_img, "Zone" + str(i+1), Center, 0, 1, (0, 0, 255), 2)
            
            Shown_img = cv2.copyMakeBorder(Shown_img, 100, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            text_0 = "Counts:" + CountsLine
            text_1 = "Frame: " + str(frame_idx)
            text_2 = "Total Counts: " + str(VD.NCV + VD.NCB)
            cv2.putText(Shown_img, text_0, (5, 30),  0, 0.7, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(Shown_img, text_1, (5, 60),  0, 0.7, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(Shown_img, text_2, (5, 90),  0, 0.7, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

            
            cv2.imshow('Real-Time Video', Shown_img)
            if cv2.waitKey(1) != -1:
                break
        
        # save detection and tracking data
        if save_csv:    
            for *OBB, cls, ID in Det:
                line = (frame_idx, ID, *OBB, cls) 
                with open(csv_path + '.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(line)
        
        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, Shown_img)
                #print(f" The image with the result is saved in: {save_path}\n")
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w, h = im0s.shape[1], im0s.shape[0]
                    else:  # stream
                        fps, w, h = 10, im0s.shape[1], im0s.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(Shown_img)
        
    if save_csv or save_img:
        print(f"\nResults saved to {save_dir}")
        
    if opt.LearnPatterns:
        if VD.TrainingMode:
            print("\nUnable to meet the target number of tracks for training. Proceeding with the available tracks")
            VD.CollectData.ReadyForTraining = True
            VD.Learn_Patters()
            
    print("\nTotal number of verified tracks: ", VD.NCV )
    print("Total number of broken tracks: ", VD.NCB )
    print("Total number of unclassified broken tracks: ", VD.NNB )
    print("Counts:")
    print(VD.Counts)
    
    cv2.destroyAllWindows()
    print("Finished at:", datetime.datetime.now().strftime("%H:%M:%S"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-csv', action='store_true', help='save results to *.csv')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--LearnPatterns', action='store_true', help='Learn the movement patters')
    parser.add_argument('--TracksPerLane', type=int, default=10, help='Tracks per Lane in Data Matrix (N)')

    opt = parser.parse_args()
    print(opt)
    
    sort_tracker = Sort(max_age=10,
                       min_hits=2,
                       iou_threshold=0.1) 
    
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
