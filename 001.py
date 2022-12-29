import os,glob,cv2,shutil
import numpy as np
def mouse_click(event, x, y, flags, para):
    global one_points
    if event == cv2.EVENT_LBUTTONDOWN:  # 左边鼠标点击
        one_points.append([x, y])

if __name__ == '__main__':

    imgDir = '/home/yao/Documents/DDRNet.pytorch-main/run/detect/exp14'
    labelDir = '/home/yao/Documents/DDRNet.pytorch-main/run/detect/exp15'
    saveDir = '/home/yao/Documents/DDRNet.pytorch-main/run/detect/pick3'
    for i,im in enumerate(os.listdir(imgDir)):
        imName = im##.split('.')[0] + '.jpg'
        imPath = os.path.join(imgDir,im)
        labelPath = os.path.join(labelDir,imName.replace('jpg','png'))
        image = cv2.imread(imPath, cv2.IMREAD_COLOR)
        label = cv2.imread(labelPath,cv2.IMREAD_GRAYSCALE)

        label = np.expand_dims(label,axis=2)
        label = np.concatenate((label,label,label),axis=2)
        print(image.shape)
        print(label.shape)
        img = np.hstack((image,label))
        one_points = []
        cv2.namedWindow("img")
        cv2.setMouseCallback("img", mouse_click)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        print(len(one_points),one_points)
        if len(one_points) == 1:
            shutil.copy(labelPath,saveDir + '/' + imName)
            print('saved ',saveDir + imName)


