import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.misc as m



colors =[
        [255, 191, 0],  # 深天蓝
        # [1,1,1],  # 地板： 草地绿 [138,138,138]
        [138, 223, 152],
        [255, 248, 248],  # 墙壁：幽灵白
        [0, 165, 255],  # 椅子：橙子色
        [213, 176, 197],  # 家具：淡紫色（薰衣草色）
        [0, 255, 0],  # 其他物体：	亮绿色
        [138, 223, 152],  # 窗口：草地绿
        [0, 255, 255],  # 床：黄色（颜色与颜色表的RB值位置相反）
        [156, 158, 222],  # 窗口：薰衣草紫
        [255, 0, 0],  # 纯蓝色
        [0, 255, 0],  # 线缆：亮绿色
        [79, 79, 47],  # 深石板灰
        ]
# colors =[
#                 [1, 1, 1],  # 深天蓝
#                 [100, 100, 100],  # 地板： 草地绿

label_colours = dict(zip(range(2), colors))


def transformY(image,img_size):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    image = torch.from_numpy(image).float()
    return image


def decode_segmap(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 2):
        # if l == 1: # 为了生成label image
        #    r[temp == l]= 0  # [l][0]
        #    g[temp == l]= 0  # [l][1]
        #    b[temp == l]= 0  # [l][2]
        # else:
        r[temp == l] = label_colours[l][0]#[l][0]
        g[temp == l] = label_colours[l][1]#[l][1]
        b[temp == l] = label_colours[l][2]#[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb