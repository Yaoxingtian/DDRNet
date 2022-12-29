
import cv2
import numpy as np

from lib.datasets.base_dataset import BaseDataset
from lib.datasets.GZdata_add_ikea0124 import Cityscapes
import argparse
import os

import timeit
import glob

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image
# import _init_paths
import models
import datasets
from config import config
from config import update_config
import test_function

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="/home/yao/Documents/DDRNet.pytorch-main/experiments/cityscapes/ddrnet23.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    # img_path = "E:/vSLAM/DDRNet.pytorch-main/data/cityscapes/leftImg8bit/train/"
    # logger, final_output_dir, _ = create_logger(config, args.cfg, 'test')

    final_output_dir = 'output\\cityscapes\\ddrnet23_slim'
    # logger.info(pprint.pformat(args))
    # ogger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.' + config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)

    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pth')
        # model_state_file = os.path.join(final_output_dir, 'final_state.pth')


    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()  # 此处去掉models字样，用的还是best.pth权重
                       if k[6:] in model_dict.keys()}
    """
    cc = {}
    for k1,v1 in pretrained_dict.items():
        if k1[6:] in model_dict.keys():
            cc.update({k1[6:]:v1})          
    """

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    # gpus = [0]
    model = nn.DataParallel(model, device_ids=gpus).cuda()  # 数据并行
    model.eval()
    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,

        
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1)
    x = 1
    cc = BaseDataset()

    WSI_MASK_PATH = 'G:/dataset/150_left/'

    paths = glob.glob(os.path.join(WSI_MASK_PATH, '*'))
    # bb = Cityscapes(config.DATASET.ROOT,config.DATASET.TEST_SET)
    cap = cv2.VideoCapture('G:/googleDownload/20200916/test.avi')
    save_pathh = "F:/dataset/save_path/20211221_04/"
    # cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    cot = 15
    while (True):
        """
              camera_matrix = np.array(
                  [[215.75641069759234, 0.000000, 319.5268713991538], [0.000000, 286.21109748598116, 237.74270934744212],
                   [0, 0, 1]], dtype=np.float32)
              distortion = np.array([0.1665615720733058, 0.06861980718197339, -0.01274480560597639, 0.004648347019391406],
                                    dtype=np.float32)
              map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, distortion, np.eye(3), camera_matrix,
                                                               (640, 480), cv2.CV_32FC1)
              frame1 = cv2.remap(frame, map1, map2, cv2.INTER_NEAREST)
              #image2 = cv2.imread(img_path+'00000001.png', cv2.IMREAD_COLOR)
              """
        ret, frame = cap.read()

        image2 = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        img = image2
        frame = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        images1 = np.array(frame, dtype=np.uint8)
        img_size = (480, 640)
        images1 = test_function.transformY(images1, img_size)

        out = None
        if x > 0:
            numel = sum([x.numel() for x in images1])
            storage = images1.storage()._new_shared(numel)
            out = images1.new(storage)
        x = x - 1
        yz = (images1,)
        images = torch.stack(yz, 0, out=out)
        images = images.to("cuda")

        image2 = cc.input_transform(image2)
        image2 = image2.transpose((2, 0, 1))
        image2 = image2[np.newaxis ,:]

        image2 = torch.from_numpy(image2)

        start = timeit.default_timer()
        pred = test_dataset.multi_scale_inference(config ,model ,image2)
        end = timeit.default_timer( ) -start
        end = end * 1000
        print(end, 'ms')
        pred2 = np.squeeze(pred.data.max(1)[1].cpu().numpy(), axis=0)
        decoded = test_function.decode_segmap(pred2)
        img_input = np.squeeze(images.cpu().numpy(), axis=0)
        blend = img_input * 0.1 + decoded * 0.9
        cv2.imshow('img' ,img)
        cv2.imshow('DDRNet', blend)
        cv2.waitKey(1)
        strx = input("请输入：")


        if strx=='1':
            ptt = save_pathh+str(cot)+".png"
            cv2.imwrite(ptt,img)
            cot = cot+1
        else:
            continue
        # logger.info('Mins: %d' % np.int((end - start) / 60))
        # logger.info('Done')


if __name__ == '__main__':
    main()
