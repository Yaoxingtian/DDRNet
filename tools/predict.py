import sys,os
# sys.path.append('/home/yaoxing/DDRNet.pytorch-main/lib')
GRANDFA = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(GRANDFA) 
print(os.path.realpath(__file__))
import cv2
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
import lib.datasets
from lib.config import config
from lib.config import update_config
import test_function
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # default="/home/yao/Documents/DDRNet.pytorch-main/experiments/cityscapes/ddrnet23_slim.yaml",
                        default="/home/lyn/Documents/workspace/DDRNet.pytorch-main/experiments/cityscapes/ddrnet23_ilife.yaml",
                        # default="./experiments/cityscapes/ddrnet39.yaml",
                        type=str)
    parser.add_argument('--images',
                        help="Modify config options using the command-line",
                        default='/home/lyn/Documents/workspace/datasets/outputs/second_time/light2/samples',
                        )
    parser.add_argument('--vide',
                        help="Modify config options using the command-line",
                        default='/home/lyn/Documents/workspace/DDRNet.pytorch-main/videos/430269720-1-208.mp4',
                        )
    parser.add_argument('--mode',default='image',help="images,video",)
    parser.add_argument('--videopath',default='/home/lyn/Documents/workspace/DDRNet.pytorch-main/run/detect/test.mp4',help="images,video",)

    parser.add_argument('--save_path',default='/home/lyn/Documents/workspace/DDRNet.pytorch-main/run/detect')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def multi_scale_aug(image, label=None,
                    rand_scale=1, rand_crop=True):      #多尺度扩充
    base_size = 640
    long_size = int(base_size * rand_scale + 0.5)
    h, w = image.shape[:2]
    if h > w:
        new_h = long_size
        new_w = int(w * long_size / h + 0.5)
    else:
        new_w = long_size
        new_h = int(h * long_size / w + 0.5)

    image = cv2.resize(image, (new_w, new_h),
                       interpolation=cv2.INTER_LINEAR)
    if label is not None:
        label = cv2.resize(label, (new_w, new_h),
                           interpolation=cv2.INTER_NEAREST)
    else:
        return image
    # if rand_crop:
    #     image, label = rand_crop(image, label)
    return image, label

def inference(config, model, image, flip=False): #推断
    size = image.size()
    pred = model(image)         #推断，进行前向操作

    if config.MODEL.NUM_OUTPUTS > 1:
        pred = pred[config.TEST.OUTPUT_INDEX]
    pred = F.interpolate(
        input=pred, size=size[-2:],
        mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
    )               #上下采样
    if flip:
        flip_img = image.numpy()[:, :, :, ::-1]
        flip_output = model(torch.from_numpy(flip_img.copy()))

        if config.MODEL.NUM_OUTPUTS > 1:
            flip_output = flip_output[config.TEST.OUTPUT_INDEX]
        flip_output = F.interpolate(
            input=flip_output, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        flip_pred = flip_output.cpu().numpy().copy()
        flip_pred = torch.from_numpy(
            flip_pred[:, :, :, ::-1].copy()).to(device)#cuda()
        pred += flip_pred
        pred = pred * 0.5
    return pred.exp()

def multi_scale_inference(config, model, image, scales=[1], flip=False):
    batch, _, ori_height, ori_width = image.size()
    crop_size = (480, 640)
    num_classes = 2
    assert batch == 1, "only supporting batchsize 1."
    image = image.cpu().numpy()[0].transpose((1, 2, 0)).copy()
    stride_h = np.int32(crop_size[0] * 1.0)
    stride_w = np.int32(crop_size[1] * 1.0)
    final_pred = torch.zeros([1, num_classes,
                              ori_height, ori_width]).to(device)#cuda()
    for scale in scales:
        new_img = multi_scale_aug(image=image,
                                       rand_scale=scale,
                                       rand_crop=False)
        height, width = new_img.shape[:-1]

        if scale <= 1.0:
            new_img = new_img.transpose((2, 0, 1))
            new_img = np.expand_dims(new_img, axis=0)
            new_img = torch.from_numpy(new_img)
            preds = inference(config, model, new_img, flip)
            preds = preds[:, :, 0:height, 0:width]
        else:
            new_h, new_w = new_img.shape[:-1]
            rows = np.int(np.ceil(1.0 * (new_h -
                                         crop_size[0]) / stride_h)) + 1
            cols = np.int(np.ceil(1.0 * (new_w -
                                         crop_size[1]) / stride_w)) + 1
            preds = torch.zeros([1, num_classes,
                                 new_h, new_w]).to(device)#.cuda()
            count = torch.zeros([1, 1, new_h, new_w]).to(device)#.cuda()

            for r in range(rows):
                for c in range(cols):
                    h0 = r * stride_h
                    w0 = c * stride_w
                    h1 = min(h0 + crop_size[0], new_h)
                    w1 = min(w0 + crop_size[1], new_w)
                    h0 = max(int(h1 - crop_size[0]), 0)
                    w0 = max(int(w1 - crop_size[1]), 0)
                    crop_img = new_img[h0:h1, w0:w1, :]
                    crop_img = crop_img.transpose((2, 0, 1))
                    crop_img = np.expand_dims(crop_img, axis=0)
                    crop_img = torch.from_numpy(crop_img)
                    pred = inference(config, model, crop_img, flip)
                    preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                    count[:, :, h0:h1, w0:w1] += 1
            preds = preds / count
            preds = preds[:, :, :height, :width]

        preds = F.interpolate(
            preds, (ori_height, ori_width),
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )
        final_pred += preds
    return final_pred

def get_result(frame,model,saveDir, im_name,args):

    image_0 = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    # frame = Image.fromarray(cv2.cvtColor(image0, cv2.COLOR_BGR2RGB))
    # images1 = np.array(frame, dtype=np.uint8)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image_0.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    images1 = torch.from_numpy(image).float()
    # images1 = test_function.transformY(images1, img_size)
    x = 1
    cc = BaseDataset()
    out = None
    # numel()返回元素个数
    # torch.storage() storage可以直接把文件映射到内存中进行操作
    if x > 0:
        numel = sum([x.numel() for x in images1])
        storage = images1.storage()._new_shared(numel)
        out = images1.new(storage)
    x = x - 1
    yz = (images1,)
    images = torch.stack(yz, 0, out=out)
    images = images.to(device)  # cuda or cpu

    image2 = cc.input_transform(image_0)
    image2 = image2.transpose((2, 0, 1))
    image2 = image2[np.newaxis, :]
    image2 = torch.from_numpy(image2).to(device)#.cuda()

    # 将pth转化为pt
    # image3 = image2.to('cuda')
    # example = torch.rand(1, 3, 480, 640)
    # traced_script_module = torch.jit.trace(model_jtr, example)
    # traced_script_module.save("ddrnet23_slim.pt")

    start = timeit.default_timer()
    pred = multi_scale_inference(config, model, image2)
    end = timeit.default_timer() - start
    end = end * 1000
    print(end, 'ms')
    pred2 = np.squeeze(pred.data.max(1)[1].cpu().numpy(), axis=0)
    decoded = test_function.decode_segmap(pred2)
    img_input = np.squeeze(images.cpu().numpy(), axis=0)
    blend = img_input * 0.1 + decoded * 0.9  # *255

    tem = pred2.copy()
    image_0_copy = image_0.copy()
    image_0[tem == 1] = 70
    img_result = np.hstack((image_0_copy,image_0))
    # out1 = cv2.VideoWriter(args.videopath, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480))
    frame = np.array(blend,dtype=np.uint8)

    # out1.write(frame)

    pred2[tem==1] = 1
    pred2[tem!=1] =100

    tem = pred2.copy()
    image_0[tem == 1] = 70

    label = np.expand_dims(pred2, axis=2)
    label = np.array(label, dtype=np.uint8)
    label = np.concatenate((label, label, label), axis=2)

    img_result = np.hstack((image_0_copy, blend * 255, label))

    # cv2.imwrite('{}/{}'.format(saveDir, im_name), img_result)
    # cv2.imwrite('{}/{}'.format(saveDir, im_name), pred2)
    cv2.imwrite('{}/{}'.format(saveDir, im_name), blend * 255)


    # cv2.imshow('inference', label)
    # key = cv2.waitKey(10)

def main():
    args = parse_args()
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
        path = Path(path)  # os-agnostic
        if path.exists() and not exist_ok:
            suffix = path.suffix
            path = path.with_suffix('')
            dirs = glob.glob(f"{path}{sep}*")  # similar paths
            matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]  # indices
            n = max(i) + 1 if i else 2  # increment number
            path = Path(f"{path}{sep}{n}{suffix}")  # update path
        dir = path if path.suffix == '' else path.parent  # directory
        if not dir.exists() and mkdir:
            dir.mkdir(parents=True, exist_ok=True)  # make directory
        return path
    saveDir = str(increment_path(Path(args.save_path)/ 'exp',mkdir = True))
    print(saveDir)
    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.' + config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)
    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))
    final_output_dir = 'output\\cityscapes\\ddrnet23'
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pth')
    pretrained_dict = torch.load(model_state_file,map_location=device) #map_location=torch.device('cuda')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()  # 此处去掉models字样，用的还是best.pth权重
                       if k[6:] in model_dict.keys()}
    print('model', model_state_file)
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
    model_jtr = model.to(device)
    model_jtr.eval()
    model = nn.DataParallel(model, device_ids=gpus).to(device)#.cuda()
    model.eval()

    # prepare data
    # test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    # test_dataset = eval('lib.datasets.' + config.DATASET.DATASET)(
    #     root=config.DATASET.ROOT,
    #     list_path=config.DATASET.TEST_SET,
    #     num_samples=None,
    #     num_classes=config.DATASET.NUM_CLASSES,
    #     multi_scale=False,
    #     flip=False,
    #     ignore_label=config.TRAIN.IGNORE_LABEL,
    #     base_size=config.TEST.BASE_SIZE,
    #     crop_size=test_size,
    #     downsample_rate=1)
    x = 1
    cc = BaseDataset()
    # flag  = 'video'

    if args.mode == 'video':
        cap = cv2.VideoCapture(args.video)
        k = 1
        while cap.isOpened():
            ret,frame = cap.read()
            im_name = '{}.png'.format(k)
            k += 1
            get_result(frame, model, saveDir, im_name)

    else:

        data = glob.glob(os.path.join(args.images, '*'))
        data = sorted(data)
        for im in data:
            frame = cv2.imread(im)
            im_name = im.split('/')[-1].split('.')[0] + '.png'
            get_result(frame, model, saveDir, im_name,args)

        # image_0 = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        # # frame = Image.fromarray(cv2.cvtColor(image0, cv2.COLOR_BGR2RGB))
        # # images1 = np.array(frame, dtype=np.uint8)
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # image = image_0.astype(np.float32)[:, :, ::-1]
        # image = image / 255.0
        # image -= mean
        # image /= std
        # images1= torch.from_numpy(image).float()
        # # images1 = test_function.transformY(images1, img_size)
        #
        # out = None
        # # numel()返回元素个数
        # #torch.storage() storage可以直接把文件映射到内存中进行操作
        # if x > 0:
        #     numel = sum([x.numel() for x in images1])
        #     storage = images1.storage()._new_shared(numel)
        #     out = images1.new(storage)
        # x = x - 1
        # yz = (images1,)
        # images = torch.stack(yz, 0, out=out)
        # images = images.to("cpu") # cuda
        #
        # image2 = cc.input_transform(image_0)
        # image2 = image2.transpose((2, 0, 1))
        # image2 = image2[np.newaxis,:]
        # image2 = torch.from_numpy(image2)
        #
        # #将pth转化为pt
        # #image3 = image2.to('cuda')
        # #example = torch.rand(1, 3, 480, 640)
        # #traced_script_module = torch.jit.trace(model_jtr, example)
        # #traced_script_module.save("ddrnet23_slim.pt")
        #
        # start = timeit.default_timer()
        # pred = multi_scale_inference(config,model,image2)
        # end = timeit.default_timer()-start
        # end = end * 1000
        # print(end, 'ms')
        # pred2 = np.squeeze(pred.data.max(1)[1].cpu().numpy(), axis=0)
        # decoded = test_function.decode_segmap(pred2)
        # img_input = np.squeeze(images.cpu().numpy(), axis=0)
        # blend = img_input * 0.1 + decoded * 0.9 # *255
        #
        # tem = pred2.copy()
        # pred2[tem!=1] = 100
        # # cv2.imwrite('{}/{}'.format(saveDir, im_name.replace('jpg','png')),pred2)
        # cv2.imwrite('{}/{}'.format(saveDir,im_name),blend*255)
        # cv2.imshow('inference',blend)
        # key = cv2.waitKey(10)
        # # video = cv2.VideoWriter('/home/yao/Documents/DDRNet.pytorch-main/run/detect/test.mp4',
        # #                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (640, 480))
        # # blend = np.array(blend*255,dtype=np.uint8)
        # # video.write(blend)
        # # videoWriter = cv2.VideoWriter('./result.mp4', f, video_fps, (video_width, video_height))
        # # print(saveDir,im_name)
        #
        # # cv2.imshow('Img_DDRNet',img)
        # # cv2.imshow('DDRNet', blend)
        # # cv2.waitKey(0)
        # # k += 1


if __name__ == '__main__':
    main()
