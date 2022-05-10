import xml.etree.ElementTree as ET
import os
from os.path import join

sets = [('2012', 'train'),('2012', 'val'), ('2007', 'train'),('2007', 'val'), ('2007', 'test')] 

classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
"diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id,image_set):
    try:
        in_file = open('/root/mid-term/data/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
        out_file = open('/root/mid-term/yolov3/data/labels/VOC%s/%s/%s.txt'%(year, image_set,image_id), 'w')
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        # print image_id
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    except:
        out_file = open('/root/mid-term/yolov3/data/labels/VOC%s/%s/%s.txt'%(year, image_set,image_id), 'w')
        out_file.close()

    

for year, image_set in sets:
    if not os.path.exists('/root/mid-term/yolov3/data/labels/VOC%s/%s/'%(year,image_set)):
        os.makedirs('/root/mid-term/yolov3/data/labels/VOC%s/%s/'%(year,image_set))
    image_ids = open('/root/mid-term/data/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('/root/mid-term/data/%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('/root/mid-term/data/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(year, image_id))
        convert_annotation(year, image_id,image_set)
    list_file.close()
#以上部分会生成对应的 年份_train.txt,年份_val.txt多个文件，我们做算法开法，通常有训练和验证就可以了
strs_train = 'cat '+ ' '.join(['/root/mid-term/data/'+a+'_'+b+'.txt' for a,b in sets if b=='train']) +'> /root/mid-term/data/train.txt'
strs_val = 'cat '+ ' '.join(['/root/mid-term/data/'+a+'_'+b+'.txt' for a,b in sets if b=='val']) +'> /root/mid-term/data/val.txt'
strs_test = 'cat '+ ' '.join(['/root/mid-term/data/'+a+'_'+b+'.txt' for a,b in sets if b=='test']) +'> /root/mid-term/data/test.txt'
os.system(strs_train)
os.system(strs_val)
os.system(strs_test)
print("all Done!")
