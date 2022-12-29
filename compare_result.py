import os

import cv2
import numpy as np

rootImg = '/home/yaoxing/DDRNet.pytorch-main/inference_ori'
data1 = '/home/yaoxing/DDRNet.pytorch-main/inference361'
data2 = '/home/yaoxing/DDRNet.pytorch-main/inference722'
data3 = '/home/yaoxing/DDRNet.pytorch-main/inference1444'

#rootImg = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/ori_test/seg_test'
#data1 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/add361Test/seg_test'
#data2 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/add722/seg_test'
#data3 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/0124_1444pic/seg_test'

# ikea_data5_23 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/0124_1444pic/ikea_data5'
# cls6test_23 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/0124_1444pic/cls6test'
# seg_test23 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/0124_1444pic/seg_test'

# ikea_data5_39 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/39test/ikea_data5'
# cls6test_39 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/39test/cls6test'
# seg_test39 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/39test/segtest'

imgLists = os.listdir(rootImg)

for img in imgLists:
    rootImg_path = os.path.join(rootImg,img)
    data1_path = os.path.join(data1, img)
    data2_path = os.path.join(data2, img)
    data3_path = os.path.join(data3, img)

    img0 = cv2.imread(rootImg_path)
    cv2.putText(img0, 'img_org', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    img1 = cv2.imread(data1_path)
    cv2.putText(img1, 'img+361', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    img2 = cv2.imread(data2_path)
    cv2.putText(img2, 'img+722', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    img3 = cv2.imread(data3_path)
    cv2.putText(img3, 'img+1444', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    
    imgMerge1 = np.hstack((img0,img1))
    imgMerge2 = np.hstack((img2,img3))
    imgMerge = np.vstack((imgMerge1,imgMerge2))
    
    cv2.imwrite('{}/{}'.format('/home/yaoxing/DDRNet.pytorch-main/compare/ikeadata5',img),imgMerge)
    # cv2.imshow('im',imgMerge)
    # cv2.waitKey(0)


