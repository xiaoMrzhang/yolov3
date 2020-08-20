import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import csv

from utils.utils import xyxy2xywh, xywh2xyxy
from utils.mio import load_string_list, encode_label

# The annotation of KITTI
# #Values    Name      Description
# ----------------------------------------------------------------------------
#    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
#                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                      'Misc' or 'DontCare'
#    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
#                      truncated refers to the object leaving image boundaries
#    1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                      0 = fully visible, 1 = partly occluded
#                      2 = largely occluded, 3 = unknown
#    1    alpha        Observation angle of object, ranging [-pi..pi]
#    4    bbox         2D bounding box of object in the image (0-based index):
#                      contains left, top, right, bottom pixel coordinates
#    3    dimensions   3D object dimensions: height, width, length (in meters)
#    3    location     3D object location x,y,z in camera coordinates (in meters)
#    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
#    1    score        Only for results: Float, indicating confidence in
#                      detection, needed for p/r curves, higher is better.

cls_dict = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
}

class LoadKITTIImageLabel(Dataset):
    def __init__(self, path, img_size=416, batch_size=16, augment=False, is_train=True, 
                 image_size=(384,1280), out_parms=9, cache_images=False, single_cls=False, pad=0.0):
        
        if is_train:
            self.path = os.path.join(path, "training")
            train_txt_file = os.path.join(self.path, "ImageSets/train.txt")
        else:
            # self.path = os.path.join(path, "testing")
            self.path = os.path.join(path, "training")
            train_txt_file = os.path.join(self.path, "ImageSets/trainval.txt")
        self.image_dir = os.path.join(self.path, "image_2")
        self.label_dir = os.path.join(self.path, "label_2")
        self.calib_dir = os.path.join(self.path, "calib")

        if not os.path.exists(train_txt_file):
            print(train_txt_file, "is not exists, Please check the kitti path")
        self.files = load_string_list(train_txt_file)
        self.len = len(self.files)
        self.classes = ["Car", "Cyclist", "Pedestrian"]
        self.n = self.len

        # image cache
        self.imgs = [None] * self.len
        # label cache
        self.labels = [np.zeros((0, 9), dtype=np.float32)] * self.len
        self.is_train = True
        self.image_weights = False

        self.flip_prob = 0.0
        self.aug_prob = 0.0
        self.input_height = image_size[0]
        self.input_width = image_size[1]
        self.out_parms = out_parms

    def __getitem__(self, index):
        img = self.load_image(index)
        anns, K = self.load_annotations(index)

        # Change h, w -> w, h
        size = np.array([i for i in img.shape[:-1]], np.float32)[::-1]
        center = np.array([i/2 for i in img.shape[:-1]], np.float32)[::-1]
        # return img, K

        """
        resize, horizontal flip, and affine augmentation are performed here.
        since it is complicated to compute heatmap w.r.t transform.
        """

        flipped = False
        if (self.is_train) and (np.random.rand() < self.flip_prob):
            flipped = True
            img = cv2.flip(img, 1)
            center[0] = size[0] - center[0] - 1
            K[0, 2] = size[0] - K[0, 2] - 1

        affine = False
        if (self.is_train) and (np.random.rand() < self.aug_prob):
            img, target, trans_mat = random_affine(img, degrees=0, translate=.1, scale=.1)
            affine = True
            '''
            TODO: affine the label mat
            point = affine_transform(point, trans_mat)
            box2d[:2] = affine_transform(box2d[:2], trans_mat)
            box2d[2:] = affine_transform(box2d[2:], trans_mat)
            
            TODO:There is something wrong when clip after resize
            box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.input_width - 1)
            box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.input_height - 1)
            '''

        resize = False
        if img.shape[0] != self.input_height | img.shape[1] != self.input_width:
            img, ratio, pad = resize_image_with_pad(img, (self.input_height, self.input_width))
            resize = True
        
        labels = np.zeros((len(anns), 9))
        for i, a in enumerate(anns):
            a = a.copy()
            _cls = a["label"]

            locs = np.array(a["locations"])
            rot_y = np.array(a["rot_y"])
            if flipped:
                locs[0] *= -1
                rot_y *= -1
            
            # We can get 2D bbox by labels or calculate by camera&3D bbox directly
            point, box2d, box3d = encode_label(
                K, rot_y, a["dimensions"], locs
            )
            
            # 当图像中的物体不全时，计算出的2D框会超出图像大小范围，这里先使用标注信息替代
            box2d = a["bbox"]
            labels[i, 0] = _cls
            labels[i, 1:5] = np.array(box2d)
            labels[i, 5:8] = np.array(a["dimensions"])
            labels[i, 8] = rot_y
            # h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]
        
        nL = len(labels)
        if nL > 0:
            if resize:
                labels[:, 1] = ratio[0] * labels[:, 1] + pad[0]  # pad width
                labels[:, 2] = ratio[1] * labels[:, 2] + pad[1]  # pad height
                labels[:, 3] = ratio[0] * labels[:, 3] + pad[0]
                labels[:, 4] = ratio[1] * labels[:, 4] + pad[1]
            
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

            #
            labels[labels < 0] = 0.0
        label_out = torch.zeros((nL, self.out_parms+1))
        if nL > 0:
            label_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        shapes = (size[1], size[0]), (size[1]/img.shape[2], size[0]/img.shape[1], pad)
        return torch.from_numpy(img), label_out[:, :6], self.files[index], shapes

    def __len__(self, ):
        return self.len

    def load_image(self, index):
        # load 1 image from dataset
        img = self.imgs[index]
        if img is None:  # not in cache
            image_file_path = os.path.join(self.image_dir, self.files[index]+".png")
            img = cv2.imread(image_file_path)
            assert img is not None, "Image Not Find" + image_file_path
        return img

    
    def load_annotations(self, idx):
        annotations = []
        file_name = self.files[idx]
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']

        if self.is_train:
            with open(os.path.join(self.label_dir, file_name+".txt"), 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)

                for line, row in enumerate(reader):
                    if row["type"] in self.classes:
                        annotations.append({
                            "class": row["type"],
                            "label": cls_dict[row["type"]],
                            "truncation": float(row["truncated"]),
                            "occlusion": float(row["occluded"]),
                            "alpha": float(row["alpha"]),
                            "bbox":[float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])],
                            "dimensions": [float(row['dl']), float(row['dh']), float(row['dw'])],
                            "locations": [float(row['lx']), float(row['ly']), float(row['lz'])],
                            "rot_y": float(row["ry"])
                        })
        
        # get camera intrinsic matrix K
        with open(os.path.join(self.calib_dir, file_name+".txt"), 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                if row[0] == 'P2:':
                    K = row[1:]
                    K = [float(i) for i in K]
                    K = np.array(K, dtype=np.float32).reshape(3, 4)
                    K = K[:3, :3]
                    break

        return annotations, K

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets, M

def affine_transform(point, matrix):
    point_exd = np.array([point[0], point[1], 1.])
    new_point = np.matmul(matrix, point_exd)

    return new_point[:2]

def resize_image_with_pad(img, new_shape, color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r_h = new_shape[0] / shape[0]
    r_w = new_shape[1] / shape[1]
    r = min(r_h, r_w)

    #Computer pad
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding


    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh) 

def reisze_with_pad(img, new_shape, color=(114, 114, 114)):
    pass

