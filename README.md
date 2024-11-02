# Fisheye-Traffic-Counts

A Vehicle Movement Counting System for Intersections Monitored by an Overhead Fisheye Camera

Implementation of paper - [Enhanced Vehicle Movement Counting at Intersections via a Self-Learning Fisheye Camera System] [Link](https://ieeexplore.ieee.org/abstract/document/10542980)

https://github.com/user-attachments/assets/153a8c6a-8e1e-4922-b575-68ece9efd61e


# Installation
``` shell
cd code_directory
pip install -r requirements.txt
```




Download the preferred [YOLOv7 ](https://github.com/WongKinYiu/yolov7) weights to the code directory.

[`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) [`yolov7x.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) [`yolov7-w6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) [`yolov7-e6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) [`yolov7-d6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) [`yolov7-e6e.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)

# Configuration
1. Run config.py and select one of the images from your overhead fisheye dataset to generate a cfg.txt file and define zones for the intersection. The sample configuration files are provided in the 'Counter/cfg' directory for the images in the 'inference/images' folder in this directory.

2. Open 'Counter/cfg/cfg.txt' and define the number of lanes for each street.

# Training
The algorithm can operate without prior training, but for improved accuracy in counting broken tracks, it is recommended to initially run it in training mode. This allows the system to extract traffic patterns (Black paths in the video) at the intersection, enabling more accurate detection of broken tracks and enhancing overall counting precision.

On video:
``` shell
python DTC.py --weights yolov7-e6e.pt --conf 0.25 --no-trace --img-size 1280 --source inference/yourvideo.mp4 --LearnPatterns --TracksPerLane 50
```
On image:
``` shell
python DTC.py --weights yolov7-e6e.pt --conf 0.25 --no-trace --img-size 1280 --source inference/images_folder --LearnPatterns --TracksPerLane 50
```

# Inference
On video:
``` shell
python DTC.py --weights yolov7.pt --conf 0.25 --no-trace --view-img --img-size 640 --source yourvideo.mp4
```
On image:
``` shell
python DTC.py --weights yolov7.pt --conf 0.25 --no-trace --view-img --img-size 640 --source inference/images_folder
```
