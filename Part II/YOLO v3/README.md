# NeuNetOne --- Part II YOLO v3

Neural Network &amp; Deep Learning Midtern Homework (Part II): YOLO v3

## Introduction

YOLO v3 model for object detection on VOC dataset.

## Usage

1. down the dataset of VOC. Note that we use VOC2007(http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html), and VOC 2012(http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html). 

2. Run `python convert.py` to convert the label format of VOC dataset to that of we need in YOLO, and `python moveimage.py` to move the images and labels to the required directory. Specificly, we need labels and images are stored as follow and change the corresponding voc.yaml.

   ```{plain}
   data
   ├── images
   └── labels
   ```


3. Run `python train.py --batch 16 --epochs 200 --data voc.yaml --weights yolov3.pt` to train models with pretrained weights. The results will be shown in  runs/train/exp

4. Run `python val.py --data voc.yaml --weights (the place where you put your weights) --iou 0.65 --half --task test` to validate the model with the test dataset. The results will be shown in  runs/val/exp




```

## Author

Xiaoyi Zhu, [18307130047](mailto:18307130047@fudan.edu.cn)
