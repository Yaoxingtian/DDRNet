# coding=utf-8
# 鼠标左键按下拖动鼠标画矩形方框，按下‘m’左键按下记录左键轨迹；打出红点；
import cv2
import numpy as np
points = [(38.90756302521005, 421.9327731092437), (0.0, 347.3599439775911)]
# img= cv2.imread('/home/lyn/Documents/workspace/datasets/outputs/a carpet on the floor/good/00005.png')
# cv2.circle(img,center=points,radius=10,color=(0,0,255))
# cv2.imshow('img',img)
# cv2.waitKey(0)
# print(img.size)
# print(img.shape)

if len(points) == 2:
    a =1
else:
    a = 22

print(a)