import sys,os
# sys.path.append('/home/yaoxing/DDRNet.pytorch-main/lib')
GRANDFA = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(GRANDFA) 
print(os.path.realpath(__file__))
import cv2
import numpy as np
from lib.datasets.base_dataset import BaseDataset
from lib.datasets.GZdata_add_ikea0124 import Cityscapes
# from base_dataset import BaseDataset
# from cityscapes import Cityscapes
import argparse
import os
import timeit
import glob
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image
#import _init_paths
import lib.models
import lib.datasets
from lib.config import config
from lib.config import update_config
import test_function
# print('vv', eval('models'))
# exit()
print(000)
def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        #default="/home/yao/Documents/DDRNet.pytorch-main/experiments/cityscapes/ddrnet23_slim.yaml",
                        default="./experiments/cityscapes/ddrnet23_ilife.yaml",
                        # default="./experiments/cityscapes/ddrnet39.yaml",
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
    # WSI_MASK_PATH = '/home/yao/Documents/DDRNet.pytorch-main/tools/data/cityscapes/leftImg8bit/val/'
    # WSI_MASK_PATH = '/home/Data/Test/seg_test/all_images'
    #img_path = "E:/vSLAM/DDRNet.pytorch-main/data/cityscapes/leftImg8bit/train/"
    #logger, final_output_dir, _ = create_logger(config, args.cfg, 'test')

    final_output_dir = 'output\\cityscapes\\ddrnet23'
    #logger.info(pprint.pformat(args))
    #ogger.info(pprint.pformat(config))
    # save_path = "/home/yaoxing/DDRNet.pytorch-main/inference"
    # cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('lib.models.' + config.MODEL.NAME)
        # print('vv', eval('models'))
        # print('11',config.MODEL.NAME)
        # exit()
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('lib.models.' + config.MODEL.NAME + '.get_seg_model')(config)
    # print('model',model)
    # exit(1)
    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pth')
        # model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    print('model',model_state_file)

    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()  # ????????????models?????????????????????best.pth??????
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
    #model_jtr = model.to('cuda')
    model_jtr = model.to('cpu')
    model_jtr.eval()


    model = nn.DataParallel(model, device_ids=gpus).cuda()
    model.eval()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('lib.datasets.' + config.DATASET.DATASET)(
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


    # WSI_MASK_PATH = '/home/Data/Test/seg_test/all_images'
    WSI_MASK_PATH ='/home/Data/raw_data/ikea_decathlon_data/ikea_1s_split/part_5'
    # WSI_MASK_PATH = '/home/Data/Test/cls6_testdata/images'

    save_path = "/home/yaoxing/DDRNet.pytorch-main/inference"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    paths = glob.glob(os.path.join(WSI_MASK_PATH, '*'))
    #bb = Cityscapes(config.DATASET.ROOT,config.DATASET.TEST_SET)
    #cap = cv2.VideoCapture('G:/googleDownload/20200916/WIN_20210203_14_29_38_Pro.mp4')
    # cot = 1
    for path in paths:
        frame = cv2.imread(path)
        # print(cot)
        # cot = cot+1
        #if cot<17000:
       #     continue

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
        # print(path)
        # exit(0)
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
        image2 = image2[np.newaxis,:]

        image2 = torch.from_numpy(image2)
        #???pth?????????pt
        #image3 = image2.to('cuda')
        #example = torch.rand(1, 3, 480, 640)
        #traced_script_module = torch.jit.trace(model_jtr, example)
        #traced_script_module.save("ddrnet23_slim.pt")

        start = timeit.default_timer()
        pred = test_dataset.multi_scale_inference(config,model,image2)
        end = timeit.default_timer()-start
        end = end * 1000
        print(end, 'ms')
        pred2 = np.squeeze(pred.data.max(1)[1].cpu().numpy(), axis=0)
        decoded = test_function.decode_segmap(pred2)
        img_input = np.squeeze(images.cpu().numpy(), axis=0)

        blend = img_input * 0.1 + decoded * 0.9 # *255

        im_name = path.split('/')[-1]
        cv2.imwrite('{}/{}'.format(save_path,im_name),blend*255)
        print(save_path,im_name)
   
        # cv2.imshow('Img_DDRNet',img)
        # cv2.imshow('DDRNet', blend)
        # cv2.waitKey(0)
        # strx = input("????????????")
        #
        # if strx == '1':
        #     ptt = save_pathh + str(cot) + ".png"
        #     cv2.imwrite(ptt, img)
        #     cot = cot + 1
        # else:
        #     continue
        #logger.info('Mins: %d' % np.int((end - start) / 60))
        #logger.info('Done')


if __name__ == '__main__':
    main()
