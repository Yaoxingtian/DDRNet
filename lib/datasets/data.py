# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
import os,yaml
import cv2,random
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .base_dataset import BaseDataset
from config import config
# from lib.config import config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = ['cuda' if torch.cuda.is_available() else 'cpu']
def load_mosaic(index,flag):

    data=yaml.load(open('/home/yaoxing/DDRNet.pytorch-main/experiments/cityscapes/ddrnet23_ilife.yaml','r'),Loader=yaml.FullLoader)
    if flag:
        dataset = data['DATASET']['TRAIN_SET']
    else:
        dataset = data['DATASET']['TEST_SET']
    images_list_ = [line.strip().split() for line in open(data['DATASET']['ROOT'] + '/' + dataset,'r')]

    images_list = []
    labels_list = []
    for item in images_list_:
        image_path, label_path = item
        
        images_list.append(image_path)
        labels_list.append(label_path)

    images_list.sort()
    labels_list.sort()
    s = img_size = 640
    mosaic_border = [-img_size // 2, -img_size // 2]
    # print(len(images_list))
    n = len(images_list)
    
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4, segments4 = [], []
    
    # random.uniform(-640, 2 * 640 + 320)
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)  # mosaic center x, y
    indices = [index] + random.choices(range(n), k=3)  # 3 additional image indices
    random.shuffle(indices)
    # print(indices)
    for i, index in enumerate(indices):
        # Load image
        imPath = '/home/Data/Train/DDRNet_data/SUNRGBD/sunRGBD_8945' + '/' + images_list[index]
        labelPath = '/home/Data/Train/DDRNet_data/SUNRGBD/sunRGBD_8945' + '/' + labels_list[index]
        # labelPath = imPath.replace('jpg', 'png').replace('images','floor')
        # print(labelPath)
        img = cv2.imread(imPath)
        label = cv2.imread(labelPath,0)

        # label = cv2.imread(labels_list[index])
        coef = random.uniform(0.4, 1.0)
        (h, w) = img.shape[:2]
        h,w = int(h*coef),int(w*coef)
        # img, _, (h, w) = self.load_image(index)
        # place img in img4
        if i == 0:  # top left

            img4 = np.full((s * 2, s * 2, img.shape[2]), 100, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)

            label4 = np.full((s * 2, s * 2), 100, dtype=np.uint8)  # base image with 4 tiles
            x1a_label, y1a_label, x2a_label, y2a_label = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b_label, y1b_label, x2b_label, y2b_label = w - (x2a_label - x1a_label), h - (y2a_label - y1a_label), w, h
            # print(1, labelPath)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h

            x1a_label, y1a_label, x2a_label, y2a_label = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b_label, y1b_label, x2b_label, y2b_label = 0, h - (y2a_label - y1a_label), min(w, x2a_label - x1a_label), h
            # print(2, labelPath)
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)

            x1a_label, y1a_label, x2a_label, y2a_label = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b_label, y1b_label, x2b_label, y2b_label = w - (x2a_label - x1a_label), 0, w, min(y2a_label - y1a_label, h)
            # print(3, labelPath)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            x1a_label, y1a_label, x2a_label, y2a_label = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b_label, y1b_label, x2b_label, y2b_label = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            # print(4,labelPath)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        label4[y1a_label:y2a_label, x1a_label:x2a_label] = label[y1b_label:y2b_label, x1b_label:x2b_label]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b
        # print(img4.min(),img4.max(),label4.min(),label4.max(),img4.shape,label4.shape)
    img4 = cv2.resize(img4, (640, 480),interpolation=cv2.INTER_LINEAR)
    label4 = cv2.resize(label4, (640, 480),interpolation=cv2.INTER_NEAREST)
 
    # im5 = np.hstack((img4, label4))
    # cv2.imwrite('001.jpg', img4)
    # cv2.imwrite('002.png', label4)
    # cv2.imshow('im',im5)
    # cv2.waitKey(0)
    # exit()
        # Labels
    #     labels, segments = self.labels[index].copy(), self.segments[index].copy()
    #     if labels.size:
    #         labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
    #         segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
    #     labels4.append(labels)
    #     segments4.extend(segments)
    #
    # # Concat/clip labels
    # labels4 = np.concatenate(labels4, 0)
    # for x in (labels4[:, 1:], *segments4):
    #     np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    # img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    # img4, labels4 = random_perspective(img4,
    #                                    labels4,
    #                                    segments4,
    #                                    degrees=self.hyp['degrees'],
    #                                    translate=self.hyp['translate'],
    #                                    scale=self.hyp['scale'],
    #                                    shear=self.hyp['shear'],
    #                                    perspective=self.hyp['perspective'],
    #                                    border=self.mosaic_border)  # border to remove

    
    # exit()
    return img4, label4

import os,glob

class sunRGBD_8945(BaseDataset):
    # images_list = []
    # labels_list = []
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=5,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=640,
                 crop_size=(240, 320),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 train=True
                ):

        super(sunRGBD_8945, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        self.indices = len(self.files)
        self.train = train


        if num_samples:
            self.files = self.files[:num_samples]
        """
        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}
        
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539, 0.9843, 
                                        1.1116, 0.9037, 1.0865, 1.0955, 
                                        1.0865, 1.1529, 1.0507]).cuda()

        """
        self.label_mapping = {
                            0: ignore_label,
                            1: 1,
                            100: 0,
                            # 1: ignore_label,
                            2: ignore_label,
                            3: ignore_label,
                            4: ignore_label,
                            5: ignore_label }
                            # 13: ignore_label,

        self.class_weights = torch.FloatTensor([0.8373,0.918,])#.cuda()
        # self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
        #                                 1.0166, 0.9969, 0.9754, 1.0489,
        #                                 0.8786, 1.0023, 0.9539, 0.9843,
        #                                 1.1116
        #                                ]).cuda()
    def read_files(self):

        files = []

        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
                # self.images_list.append(image_path)
                # self.labels_list.append(label_path)
        # return files,images_list,labels_list
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v

        return label


    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        img_path = item["img"]
        # img_path = self.root+config.DATASET.DATASET + "/"+item["img"]
        #path1 = self.root+'GZdata_add_ikea0124' + "/"+item["img"]
        #image = cv2.imread(os.path.join(self.root,"cityscapes/",item["img"]),cv2.IMREAD_COLOR)
        # image,label = load_mosaic(index,self.train)
        # cv2.imwrite('/home/yaoxing/DDRNet.pytorch-main/run/img/{}.jpg'.format(index),image)
        # cv2.imwrite('/home/yaoxing/DDRNet.pytorch-main/run/labels/{}.png'.format(index),label)
        image = cv2.imread(img_path,cv2.IMREAD_COLOR)
        try:
            image.shape
        except:
            print('error'+img_path)
        #print(os.path.join(self.root,'cityscapes',item["img"]))
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        size = image.shape
        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        # label_path = self.root + config.DATASET.DATASET + "/"+item["label"]
        label_path = item["label"]
        # path2 = self.root+'ikea0124' + "/"+item["label"]
        #label = cv2.imread(os.path.join(self.root,"cityscapes/",item["label"]),cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        try:
            label.shape
        except:
            print('error'+label_path)
        #label1 = np.unique(label)

        label = cv2.resize(label, (640, 480), interpolation=cv2.INTER_NEAREST)

        label = self.convert_label(label)           #将label替换为需求类别
        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).to(device)#cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width),
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))
    
class cityscapes(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=5,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=640,
                 crop_size=(240, 320),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(cityscapes, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]
        self.label_mapping = {
                            0: ignore_label,
                            1: 1,
                            100: 0,
                            20: ignore_label,
        }

        # self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,1.0166, ]).cuda()
        self.class_weights = torch.FloatTensor([0.8373,0.918,])

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label


    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        path1 = item["img"]
        # print(path1)
        #image = cv2.imread(os.path.join(self.root,"cityscapes/",item["img"]),cv2.IMREAD_COLOR)
        image = cv2.imread(path1,cv2.IMREAD_COLOR)
        try:
            image.shape
        except:
            print('error'+path1)
        # print(path1)
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        size = image.shape
        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        path2 = item["label"]
        #label = cv2.imread(os.path.join(self.root,"cityscapes/",item["label"]),cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(path2,cv2.IMREAD_GRAYSCALE)
        #print(np.unique(label))
        try:
            label.shape
        except:
            print('error'+path2)
        #label1 = np.unique(label)
        #print(label1)
        #cv2.imshow("img", image)
        #cv2.imshow("label",label)
        #cv2.waitKey(0)
        label = cv2.resize(label, (640, 480), interpolation=cv2.INTER_NEAREST)

        label = self.convert_label(label)           #将label替换为需求类别
        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width), 
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )            
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))



