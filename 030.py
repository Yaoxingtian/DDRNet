import os

import cv2,glob
import numpy as np

def replace_floor_carpet(image ,label ,index):
    '''
    note: label have  3 channels
    '''
    floor_carpet_dir = '/home/Data/Train/DDRNet_data/train/floor_carpet'
    label_rgb = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
    label_rgb = cv2.resize(label_rgb ,(640 ,480))

    image  = cv2.resize(image ,(640 ,480))

    floor_list = glob.glob(floor_carpet_dir + '/*')
    n = len(floor_list)
    floor_seed = np.random.randint(0, n - 1)

    mask = np.where(label_rgb==1)
    # floor_img = cv2.imread('0001.jpg')
    floor_img = cv2.imread(floor_list[floor_seed])
    # floor_img = cv2.resize(floor_img,(640,480))
    # print(image.shape)
    # exit()
    floor_mask = floor_img[mask]
    # tem = label.copy()
    # floor_mask = floor_img[tem==1]
    image[mask] = floor_mask
    label_gray = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2GRAY)

    # cv2.imwrite('{}/{}.jpg'.format('/home/Data/Train/DDRNet_data/train/carpet_floor_aug',index),image)

    return image ,label_gray

if __name__ == '__main__':
    root = './images'
    img_list = os.listdir(root)
    for img in img_list:
        img_path = os.path.join(root,img)
        label_path = os.path.join('./labels',img.split('.')[0] + '.png')
        try:
            image = cv2.imread(img_path)
            label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        except:
            print('error:',img_path)
        replace_floor_carpet(image, label, 1)
