import os

f = open('./ddr_rknn.txt','w')
impath = '/home/yao/Documents/DDRNet.pytorch-main/images'
#labelpath = '/home/Data/Train/DDRNet_data/add_part0124/cityscapes/gtFine/train'
im_list = os.listdir(impath)
#label_list = os.listdir(labelpath)
for im in im_list:
    labelName = im.replace('jpg','png')
    f.write('/home/yao/Documents/DDRNet.pytorch-main/images/{}\n'.format(im))
    #f.write('/home/Data/Train/DDRNet_data/test/images/{} /home/Data/Train/DDRNet_data/test/labels/{}\n'.format(im,labelName))

f.close()
