import os
from os.path import join

sets = [('2012', 'train'),('2012', 'val'), ('2007', 'train'),('2007', 'val'), ('2007', 'test')] 

for year, image_set in sets:
    if not os.path.exists('/root/mid-term/yolov3/data/images/VOC%s/%s/'%(year,image_set)):
        os.makedirs('/root/mid-term/yolov3/data/images/VOC%s/%s/'%(year,image_set))
    fo = open('/root/mid-term/data/%s_%s.txt'%(year, image_set), "r")

    for line in fo.readlines():                          
        line = line.strip()
        str_mv = 'mv '+line+' '+'/root/mid-term/yolov3/data/images/VOC%s/%s/'%(year,image_set)
        os.system(str_mv)