# import sys,os
# GRANDFA = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(GRANDFA)
# print(os.path.realpath(__file__))
# import cv2
# import numpy as np
# import argparse
# import os,re
# import timeit
# import glob
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# #import _init_paths
# import models
import pprint
import logging
import numpy as np
from lib.datasets.base_dataset import BaseDataset
# from lib.datasets.data import Cityscapes
from lib.datasets.data import sunRGBD_8945
# from base_dataset import BaseDataset
from pathlib import Path
import argparse
import os,re
import timeit
import glob
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image
#import _init_paths
import models
# print('222',models)
# exit()
import datasets
from lib.config import config
from lib.config import update_config
from lib.core.function import testval, test,validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, speed_test
# CUDA_VISIBLE_DEVICES=0
import torch
# print('cuda',torch.cuda.is_available())
# exit()
def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="/home/lyn/Documents/workspace/DDRNet.pytorch-main/experiments/cityscapes/ddrnet23_ilife.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    # module = models.ddrnet_23
    # build model
    if torch.__version__.startswith('1'):
        # print('models.' + config.MODEL.NAME)
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +'.get_seg_model')(config)

    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pth')
        # model_state_file = os.path.join(final_output_dir, 'final_state.pth')      
    # logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file,map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()         #此处去掉models字样，用的还是best.pth权重
                       if k[6:] in model_dict.keys()}
    """
    cc = {}
    for k1,v1 in pretrained_dict.items():
        if k1[6:] in model_dict.keys():
            cc.update({k1[6:]:v1})          
    """
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    #gpus = [0]
    model = nn.DataParallel(model, device_ids=gpus).to(device)#cuda()      #数据并行
    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
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

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    start = timeit.default_timer()
    
    mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config, 
                                                        test_dataset, 
                                                        testloader, 
                                                        model,
                                                        sv_pred=False)

    from tensorboardX import SummaryWriter
    writer_dict = {
        'writer': SummaryWriter('/home/lyn/Documents/workspace/DDRNet.pytorch-main/run'),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    ave_loss, mean_IoU, IoU_array = validate(config, testloader, model, writer_dict)
    print('validate:','ave_loss:',ave_loss,'mean_IoU:',mean_IoU,)

    msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
        Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
        pixel_acc, mean_acc)
    logging.info(msg)
    logging.info(IoU_array)

    end = timeit.default_timer()
    # logger.info('Mins: %d' % np.int((end-start)/60))
    # logger.info('Done')


if __name__ == '__main__':
    main()

'''
DDRnet:   
MeanIU:  0.9544, 
Pixel_Acc:  0.9773, 
Mean_Acc:  0.9749, 
Class IoU: [0.96203212 0.94670672]

'''