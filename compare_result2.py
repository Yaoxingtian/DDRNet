import os

import cv2
import numpy as np

# rootImg = '/home/yaoxing/DDRNet.pytorch-main/inference_ori'
# data1 = '/home/yaoxing/DDRNet.pytorch-main/inference361'
# data2 = '/home/yaoxing/DDRNet.pytorch-main/inference722'
# data3 = '/home/yaoxing/DDRNet.pytorch-main/inference1444'

#rootImg = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/ori_test/seg_test'
#data1 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/add361Test/seg_test'
#data2 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/add722/seg_test'
#data3 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/0124_1444pic/seg_test'

ikea_data5_23 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/0124_1444pic/ikea_data5'
cls6test_23 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/0124_1444pic/cls6test'
seg_test23 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/0124_1444pic/seg_test'

ikea_data5_39 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/39test/ikea_data5'
cls6test_39 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/39test/cls6test'
seg_test39 = '/home/yaoxing/DDRNet.pytorch-main/TEST_result/39test/segtest'

ikea_data5 = os.listdir(ikea_data5_23)
cls6test = os.listdir(cls6test_23)
seg_test = os.listdir(seg_test23)
# for i in ikea_data5:
#     ikea_data5_23_path = os.path.join(ikea_data5_23,i)
#     ikea_data5_39_path = os.path.join(ikea_data5_39,i)
#     img0 = cv2.imread(ikea_data5_23_path)
#     cv2.putText(img0, 'data5_23', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
#     img1 = cv2.imread(ikea_data5_39_path)
#     cv2.putText(img1, 'data5_39', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
#     imgMerge1 = np.hstack((img0,img1))
#     # imgMerge2 = np.hstack((img2,img3))
#     # imgMerge = np.vstack((imgMerge1,imgMerge2))
    
#     cv2.imwrite('{}/{}'.format('/home/yaoxing/DDRNet.pytorch-main/compare/23vs39/ikea_data5',i),imgMerge1)

# for j in cls6test:
#     cls6test_23_path = os.path.join(cls6test_23, j)
#     cls6test_39_path = os.path.join(cls6test_39,j)
#     img2 = cv2.imread(cls6test_23_path)            
#     cv2.putText(img2, 'cls6test23', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
#     img3 = cv2.imread(cls6test_39_path)
#     cv2.putText(img3, 'cls6test39', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
#     imgMerge1 = np.hstack((img2,img3))
#     # imgMerge2 = np.hstack((img2,img3))
#     # imgMerge = np.vstack((imgMerge1,imgMerge2))
    
#     cv2.imwrite('{}/{}'.format('/home/yaoxing/DDRNet.pytorch-main/compare/23vs39/cls6test',j),imgMerge1)
    

for k in seg_test:
    seg_test_23_path = os.path.join(seg_test23, k)
    seg_test_39_path = os.path.join(seg_test39, k)

    img4 = cv2.imread(seg_test_23_path)            
    cv2.putText(img4, 'seg_test23', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    img5 = cv2.imread(seg_test_39_path)
    cv2.putText(img5, 'seg_test39', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    
    imgMerge1 = np.hstack((img4,img5))
    # imgMerge2 = np.hstack((img2,img3))
    # imgMerge = np.vstack((imgMerge1,imgMerge2))
    
    cv2.imwrite('{}/{}'.format('/home/yaoxing/DDRNet.pytorch-main/compare/23vs39/segtest',k),imgMerge1)
    # cv2.imshow('im',imgMerge)
    # cv2.waitKey(0)


