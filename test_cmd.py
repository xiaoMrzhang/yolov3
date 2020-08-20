# coding in utf-8
from utils.mio import load_string_list, save_string_list
import numpy as np
import os
import csv
import cv2
from utils import kitti_dataset
from utils.utils import xywh2xyxy
import torch

# source_path = "/home/zhangxiao/code/yolov3/coco/trainvalno5k.txt"
# dist_path = "/home/zhangxiao/code/yolov3/data/"
# select_size = 64

# coco_train_path = load_string_list(source_path)
# coco_train_path = [coco_path for coco_path in coco_train_path if "train" in coco_path.split('/')[-2]]
# select_path = np.random.choice(coco_train_path, select_size)
# dist_save_path = os.path.join(dist_path, "coco"+str(select_size)+".txt")
# save_string_list(dist_save_path, select_path)

path = "/home/zhangxiao/data/kitti"
dataset = kitti_dataset.LoadKITTIImageLabel(path, image_size=(600, 1000))
img, target, index, shapes = dataset[2431]
print(img.shape)
# img, target = kitti_dataset.random_affine(img, translate=0, shear=0)
# img, ratio, (dw, dh) = kitti_dataset.resize_image_with_pad(img, (384,1280))
img = img.permute(1,2,0)
if isinstance(img, torch.Tensor):
    img = img.numpy().astype(np.uint8)

print(img.shape, index)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for targ in target:
    x,y,w,h = targ[2:6]
    x, w = x*img.shape[1], w*img.shape[1]
    y, h = y*img.shape[0], h*img.shape[0]

    # cord = xywh2xyxy([[x,y,w,h]])
    x0 = int(x - w/2)
    y0 = int(y - h/2)
    x1 = int(x + w/2)
    y1 = int(y + h/2)

    print(x0, y0, x1, y1)
    try:
        img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
    except Exception as e:
        print(str(e))
        import pdb; pdb.set_trace()

cv2.imwrite("./test.png", img)