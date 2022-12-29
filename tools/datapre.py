import os,cv2,glob
import numpy as np


def equalizeHist_rgb(img):
    # 通道分离
    r, g, b = cv2.split(img)
    r_hist = cv2.equalizeHist(r)
    g_hist = cv2.equalizeHist(g)
    b_hist = cv2.equalizeHist(b)
    # 通道合并
    equal_out = cv2.merge((r_hist, g_hist, b_hist))
    return equal_out

def sharp(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    img_sharp = cv2.filter2D(img, -1, kernel=kernel)
    return img_sharp
 
def smooth(img):
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32)
    img_smooth = cv2.filter2D(img, -1, kernel=kernel)
    return img_smooth

if __name__ == '__main__':
    # img = '/home/yao/Documents/DDRNet.pytorch-main/error/ori/1501839277902.563218.png'
    imgRoot = '/home/yaoxing/DDRNet.pytorch-main/error/ori/good'
    saveDir = '/home/yaoxing/DDRNet.pytorch-main/error/ori/gray/good'

    for file in glob.glob('{}/*png'.format(imgRoot)):
        fileName = file.split('/')[-1]
        im = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        # im0 = equalizeHist_rgb(im)
        # im0 = sharp(im)
        # im0 = smooth(im)
        # print(type(im0))
        cv2.imwrite('{}/{}'.format(saveDir,fileName),im)
        print(file)
