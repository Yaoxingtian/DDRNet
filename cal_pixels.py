import shutil,os

# import cv2,os
# from collections import Counter
# root = '/home/lyn/Downloads/temp_data_1/labels'
# for im in os.listdir(root):
#     imPath = os.path.join(root,im)
#     img = cv2.imread(imPath,cv2.IMREAD_GRAYSCALE)
#     # cv2.imshow('im',img)
#     # cv2.waitKey(0)
#     k = [i for j in img for i in j]# 将二维数组转一维数组
#     count = Counter(k)
#     print(imPath,count)


imageDir = '/home/lyn/Documents/workspace/label1000/label_images_all'
labelDir = '/home/lyn/Documents/workspace/label1000/labels'
savePath = '/home/lyn/Documents/workspace/label1000/images'
for label in os.listdir(labelDir):
    imName1 = label.split('.png')[0] + '.jpg'
    imName2 = label.split('.png')[0] + '.png'
    print((os.path.join(imageDir,imName1)))
    if os.path.isfile(os.path.join(imageDir,imName1)):
        imPath = os.path.join(imageDir,imName1)
        shutil.copy(imPath, savePath + '/' + imName1)
    elif os.path.isfile(os.path.join(imageDir,imName2)):
        imPath = os.path.join(imageDir, imName2)
        shutil.copy(imPath, savePath + '/' + imName2)

    # shutil.copy(imPath,savePath+'/'+imName)






