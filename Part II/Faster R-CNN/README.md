# NeuNetOne --- Part II Faster R-CNN
Neural Network & Deep Learning Midtern Homework (Part II): Faster R-CNN
## Introduction
Faster R-CNN model for object detection on VOC dataset.
## Environment setting
Please see [Faster R-CNN](https://github.com/potterhsu/easy-faster-rcnn.pytorch)
## Usage
1. run `python train.py -s=voc2007 -b=resnet101 --warm_up_num_iters=1000 --warm_up_factor=0.995 --step_lr_gamma=0.99 --num_steps_to_display=1 --num_steps_to_finish=180000` to train the network.
2. run `python infer.py -s=voc2007 -b=resnet101 -c=[Your network path file] [Your image path file] [the output path]` to test your model. For example 
```
python infer.py -s=voc2007 -b=resnet101 -c=./outputs/checkpoints-20220503224625-voc2007-resnet101-c1950ad6/model-180000.pth ./test_img/000001.jpg ./00001_proposal.jpg
```
