import re
import math
import random
import numpy as np
from rknn.api import RKNN

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True, verbose_file="./log.txt")

    # pre-process config
    print('--> config model') #
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],  epochs=1,
                batch_size=1,  target_platform='rk3566',optimization_level=3
                )#reorder_channel='0 1 2',quantized_dtype="asymmetric_quantized-u8",
    print('done')

    # Load darknet model
    print('--> Loading model')
    # ret = rknn.load_pytorch(model='./yolov5x_torchscript.pt', input_size_list=[[3, 640, 640]])
    ret = rknn.load_onnx(model='/home/yao/Documents/DDRNet.pytorch-main/run/train/exp305/ddrnet23_v2.onnx')
    # ret = rknn.load_onnx(model='/home/yao/Documents/DDRNet.pytorch-main/tinyhitnet.onnx')

    if ret != 0:
        print('Load onnx model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='/home/yao/Documents/DDRNet.pytorch-main/ddr_rknn.txt')
    if ret != 0:
        print('Build pytorch failed!')
        exit(ret)
    print('done')

    # Export rknn model
    rknn.export_rknn('/home/yao/Documents/DDRNet.pytorch-main/run/train/exp305/ddrnet23_v2.rknn')

    rknn.release()
