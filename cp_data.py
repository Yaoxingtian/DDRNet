import os,shutil,cv2
import numpy as np
import matplotlib.pylab as plt
# root = '/home/lyn/Documents/pick2000/pick2/good2comp'
# dist_img = '/home/lyn/Documents/pick2000/pick2/good2'
# dist_labels = '/home/lyn/Documents/pick2000/pick2/good2labels/ori'
# for file in os.listdir(dist_img):
#     filePath = os.path.join(dist_img,file)
#     labelPath = os.path.join(dist_labels,file)
#
#     img = cv2.imread(filePath)
#     label = cv2.imread(labelPath)
#
#     conca = np.hstack((img,label))
#     cv2.imwrite('{}/{}'.format(root,file),conca)

import json,glob
save_dir = '/home/lyn/Documents/pick2000/pick2/000/labels'
root = '/home/lyn/Documents/pick2000/pick2/000/lab'
for label in glob.glob(root + '/*'):

    file = json.load(open(label,'r'))
    labels = file['shapes']
    imgName = file['imagePath']
    # imgName = images.split('/')[-1]
    img = np.zeros((480, 640, 3), np.uint8)

    color_bg = (100,100,100)
    points_bg = [(0, 0), (0, 480), (640, 480), (640, 0)]
    img = cv2.fillPoly(img, [np.array(points_bg)], color_bg)
    for i in range(len(labels)):
        points = labels[i]['points']
        new_point = []
        for p in points:
            print(p)
            mew_point = (p[0], p[1])

            point = np.asarray(mew_point).astype('int32')
            new_point.append(point)
        img = cv2.fillPoly(img, [np.array(new_point, dtype=int)], (1, 1, 1))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    save_path = os.path.join(save_dir,imgName.replace('jpg','png'))
    cv2.imwrite(save_path, img)
    # plt.imshow(img)
    # plt.show()