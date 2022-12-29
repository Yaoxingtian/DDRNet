# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 2022
@author: suiyingy
"""

import cv2
import numpy as np
import os
import json
from collections import defaultdict
'''
coco json format
to parse coco json into picture
'''

# import matplotlib.pyplot as plt

def cocojson2png(coco_dir, json_path='result.json', cls_type='part_0', save_dir='res/'):
    save_path = os.path.join(save_dir, cls_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    annotation_file = os.path.join(coco_dir, 'annotations', json_path)
    with open(annotation_file, 'r', encoding='utf-8') as annf:
        annotations = json.load(annf)
        images = [i['id'] for i in annotations['images']]

    img_anno = defaultdict(list)
    for anno in annotations['annotations']:
        for img_id in images:
            if anno['image_id'] == img_id:
                img_anno[img_id].append(anno)
    imgid_file = {}
    for im in annotations['images']:
        imgid_file[im['id']] = im['file_name']

    for img_idx in img_anno:
        imName = imgid_file[img_idx].split('/')[-1]
        impath = coco_dir + '/images/' + cls_type + '/' + imName
        image = cv2.imread(impath)
        print(impath)
        # exit(00)
        h, w, _ = image.shape
        instance_png = np.zeros((h, w), dtype=np.uint8)
        for idx, ann in enumerate(img_anno[img_idx]):
            im_mask = np.zeros((h, w), dtype=np.uint8)
            mask = []
            for an in ann['segmentation']:
                ct = np.expand_dims(np.array(an), 0).astype(int)
                contour = np.stack((ct[:, ::2], ct[:, 1::2])).T
                mask.append(contour)
            imm = cv2.drawContours(im_mask, mask, -1, 1, -1)
            imm = imm * (1000 * anno['category_id'] + idx)
            instance_png = instance_png + imm
            instance_png = np.clip(instance_png, 0, 255)
        instance_png = np.expand_dims(instance_png, axis=2).repeat(3, axis=2).astype(np.uint8)

        print(instance_png.shape)
        # print(imgid_file[img_idx].split('.')[-1])
        # exit()
        cv2.imwrite(os.path.join(save_path, imName), instance_png)


#        plt.imshow(instance_png)
#        break


if __name__ == '__main__':
    # coco_dir = "coco/"
    coco_dir = '/home/yao/Downloads/project-2-at-2022-05-30-03-49-4d98caff'
    json_path = 'result.json'
    save_path = '/home/yao/Downloads/project-2-at-2022-05-30-03-49-4d98caff/labels'
    cocojson2png(coco_dir, json_path=json_path,save_dir = save_path)