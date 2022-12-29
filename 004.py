# import os,cv2,glob
# import shutil
# root = '/home/lyn/Documents/seg-test/left'
# saveDir = '/home/lyn/Documents/seg-test/images'
# for im in os.listdir(root):
#     imPath = os.path.join(root,im)
#
#     # if os.path.getsize(imPath) == 0:
#     #     shutil.move(imPath,'/home/lyn/Documents/seg-test/error/{}'.format(im))
#     #     print(imPath)
#     # else:
#     print(imPath)
#     img = cv2.imread(imPath)
#     h,w,_ = img.shape
#     newim = img[0:480,0:640]
#     newim2 = img[0:480,640:1280]
#     # print(newim2.shape)
#     # newim2 = img[480:960, 640:1280]
#     # cv2.imwrite('{}/{}'.format('/home/yao/Documents/testdata/test',im),newim)
#     # cv2.putText(newim2, 'PIDnet 96.82', (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
#     cv2.imwrite('{}/{}'.format(saveDir,im),newim2)
#     # cv2.imshow('im',newim2)
#     # cv2.waitKey(0)
#     # print(w,h)
#
import cv2,os
import numpy as np
ddr = '/home/lyn/Documents/workspace/DDRNet.pytorch-main/run/detect/exp61'
pid = '/home/lyn/Documents/seg-test/pid'
ori = '/home/lyn/Documents/seg-test/images'
saveDir = '/home/lyn/Documents/seg-test/result'

for img in os.listdir(ori):
    ddr_im = cv2.imread(os.path.join(ddr,img))
    pid_im = cv2.imread(os.path.join(pid,img))
    im = cv2.imread(os.path.join(ori,img))
    im0 = np.hstack((im,pid_im))
    # print(os.path.join(ddr,img))
    # print(ddr_im.shape)
    im1 = np.hstack((im,ddr_im))
    cv2.putText(im1, 'DDR exp349 ', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(im0, 'PID', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # print(im1.shape)
    # print(pid_im.shape)
    # print()
    im2 = np.vstack((im1,im0))
    cv2.imwrite('{}/{}'.format(saveDir,img),im2)
    # cv2.imshow(img,im2)
    # cv2.waitKey(0)
