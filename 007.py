import numpy as np
# import cv2
# from collections import Counter
# im = cv2.imread('/home/lyn/Documents/workspace/datasets/test/ori/00209.png',cv2.IMREAD_GRAYSCALE)
# k = [i for j in im for i in j]
# count = Counter(k)
# print(count)
# cv2.imshow('im',im)
# cv2.waitKey(0)

# class Geese():
#     """大雁类"""
#
#     def __init__(self,x):
#         print("我是大雁类",x)
#
# wildGoose = Geese
#
# print(Geese(1))

import torch
import torch.nn as nn


class Module():
    def __init__(self):
        super().__init__()
        # ......

    def forward(self, x):
        # ......
        return x


data = torch.Tensor([1,1,1,1])  # 输入数据

# 实例化一个对象

model = Module()

print(model)
# 前向传播

