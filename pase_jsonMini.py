# -*- coding: utf-8 -*-
import json
import cv2
import numpy as np
import os
import shutil
import argparse


def cvt_one(json_path, img_root, save_dir, label_color):
    # load img and json
    data = json.load(open(json_path,encoding='gbk'))
    num_images = len(data)
    for i in range(num_images):
        imgName = data[i]['image'].split('/')[-1]
        img_path = os.path.join(img_root,imgName)
        img0= cv2.imread(img_path)
        print(img_path)
        # cv2.imshow('im2',img0)
        # get background data
        img_h = 480#data['imageHeight']
        img_w = 640#data['imageWidth']
        color_bg = (100,100,100)
        points_bg = [(0, 0), (0, img_h), (img_w, img_h), (img_w, 0)]
        img = cv2.fillPoly(img0, [np.array(points_bg)], color_bg)
        # cv2.imshow('im2', img)
        # draw roi
        # print(data[i])
        if 'label' in data[i]:

            labels = data[i]['label']

        # print(len(labels))
        # exit(00)
        for i in range(len(labels)):
            # name = data['shapes'][i]['label']
            points = labels[i]['points']
            new_point = []
            for point in points:
                point_x = point[0] * 6.4
                point_y = point[1] * 4.8
                mew_point = (point_x, point_y)
                point = np.asarray(mew_point).astype('int32')
                new_point.append(point)
            # color =  data['shapes'][i]['fill_color']
            # data['shapes'][i]['fill_color'] = label_color[name]  # 修改json文件中的填充颜色为我们设定的颜色
            if label_color:
                img = cv2.fillPoly(img, [np.array(new_point, dtype=int)], (1,1,1))
            else:
                img = cv2.fillPoly(img, [np.array(new_point, dtype=int)], (1,1,1))
            # print('num points:',len(new_point))
            '''
            for point in new_point:
                point_x = point[0]*6.4
                point_y = point[1]*4.8
                mew_point = (point_x,point_y)
                point = np.asarray(mew_point).astype('int32')
                cv2.circle(img, point, 10, (0, 0, 255))
            '''
            # po = np.array(points[0]).astype('int32')
            # cv2.drawMarker(img, po, (255, 255, 255), 2)
            im = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            print('im', im.shape)
            save_path = os.path.join(save_dir,imgName.replace('jpg','png'))
            cv2.imwrite(save_path, im)
            # cv2.imshow('im',img)
            # cv2.waitKey(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--label', default="/home/lyn/Documents/workspace/label1000/labels")
    parser.add_argument('--images',default='/home/lyn/Documents/workspace/label1000/label_images_all')
    parser.add_argument('--jsonfile', default='/home/lyn/Documents/workspace/label1000/project-31-at-2022-10-29-10-05-e5a9a187.json')

    args = parser.parse_args()

    save_dir = args.label
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img_root = args.images
    label_color = { #设定label染色情况
        'floor': (1,1,1)
    }
    json_path = args.jsonfile
    cvt_one(json_path, img_root, save_dir, label_color)
