# Fisheye-Traffic-Counts

A Vehicle Movement Counting System for Intersections Monitored by an Overhead Fisheye Camera

Implementation of paper - [Enhanced Vehicle Movement Counting at Intersections via a Self-Learning Fisheye Camera System] [Link](https://ieeexplore.ieee.org/abstract/document/10542980) with changes in the detection module. 



https://github.com/user-attachments/assets/26ebceb1-7aab-4727-9450-9d5894e0246d

Note: The algorithm automatically learns the movement patterns (black trajectories) by running in the training mode. 




# Installation
``` shell
cd code_directory
pip install -r requirements.txt
```




Download the preferred fine-tuned yolov7 weights for fisheye images to the code directory.

[`yolov7-fisheye.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)
[`yolov7e6e-fisheye.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)

You may want to see the fine-tuning technique used to re-train the wights here: [Fine-tuned-YOLOv7-for-Fisheye](https://github.com/MortezaAdl/Fine-tuned-YOLOv7-for-Fisheye). 

# Configuration
1. Run config.py and select one of the images from your overhead fisheye dataset to generate a cfg.txt file and define zones for the intersection. The sample configuration files are provided in the 'Counter/cfg' directory for the images in the 'inference/images' folder in this directory.

2. Open 'Counter/cfg/cfg.txt' and define the number of lanes for each street.

# Training
The algorithm can operate without prior training, but for improved accuracy in counting broken tracks, it is recommended to initially run it in training mode. This allows the system to extract traffic patterns (Black paths in the video) at the intersection, enabling more accurate detection of broken tracks and enhancing overall counting precision.

On video:
``` shell
python DTC.py --weights yolov7e6e-fisheye.pt --conf 0.25 --no-trace --view-img --img-size 1280 --source inference/yourvideo.mp4 --LearnPatterns --TracksPerLane 50
```
On image:
``` shell
python DTC.py --weights yolov7e6e-fisheye.pt --conf 0.25 --no-trace --view-img --img-size 1280 --source inference/images_folder --LearnPatterns --TracksPerLane 50
```

# Inference
On video:
``` shell
python DTC.py --weights yolov7-fisheye.pt --conf 0.25 --no-trace --view-img --img-size 640 --source inference/yourvideo.mp4
```
On image:
``` shell
python DTC.py --weights yolov7-fisheye.pt --conf 0.25 --no-trace --view-img --img-size 640 --source inference/images_folder
```
