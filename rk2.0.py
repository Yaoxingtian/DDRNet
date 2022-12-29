import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

# ONNX_MODEL = '/home/yao/Documents/DDRNet.pytorch-main/tools/output/ddrnet23.onnx'
# RKNN_MODEL = '/home/yao/Documents/DDRNet.pytorch-main/tools/output/ddrnet23_3566.rknn'

ONNX_MODEL = '/home/yao/Documents/DDRNet.pytorch-main/tinyhitnet.onnx'
RKNN_MODEL = '/home/yao/Documents/DDRNet.pytorch-main/tinyhitnet3566.rknn'
# IMG_PATH = './bus.jpg'
DATASET = '/home/yao/Documents/DDRNet.pytorch-main/coco_data.txt'
if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 225]],
                output_tensor_type='int8',
                target_platform='rk3566')
    ## ddrnet
    # rknn.config(mean_values=[[0.485, 0.456, 0.406]],
    #             std_values=[[0.229, 0.224, 0.225]],
    #             output_tensor_type='int8',
    #             target_platform='rk3566')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')