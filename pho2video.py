import os
import cv2,glob
from tqdm import tqdm
import argparse
file_dir = './exp80'
video_path = './test.avi'
imgList = glob.glob(file_dir + '/*')
imgList.sort()
# VideoWriter是cv2库提供的视频保存方法，将合成的视频保存到该路径中
# 'MJPG'意思是支持jpg格式图片
# fps = 5代表视频的帧频为5，如果图片不多，帧频最好设置的小一点
# (1280,720)是生成的视频像素1280*720，一般要与所使用的图片像素大小一致，否则生成的视频无法播放
# 定义保存视频目录名称和压缩格式，像素为1280*720
video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))
# print(list)
for im in tqdm(imgList):
    # 读取图片
    img = cv2.imread(im)
    # resize方法是cv2库提供的更改像素大小的方法
    # 将图片转换为1280*720像素大小
    # img = cv2.resize(img, (640, 720))
    # 写入视频
    video.write(img)

# 释放资源
video.release()
