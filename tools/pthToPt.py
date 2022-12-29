import models
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from lib.config import config
from config import update_config
from config import config
from config import update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="/home/yao/Documents/DDRNet.pytorch-main/experiments/cityscapes/ddrnet23_slim.yaml",
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

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        # print('vv',eval('models'))
        # print('11',eval('models.' + 'ddrnet_23_slim'))
        # exit()
        # module = eval('models.' + config.MODEL.NAME)
        module = eval('models.' + config.MODEL.NAME)
        # print('11',module)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)
    #seg_hrnet= config.MODEL.NAME

    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        #model_state_file = os.path.join(final_output_dir, 'best.pth')
        model_state_file ='/home/yao/Documents/DDRNet.pytorch-main/my/DDRNet23s_imagenet.pth'
        # model_state_file = os.path.join(final_output_dir, 'final_state.pth')


    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()  # 此处去掉models字样，用的还是best.pth权重
                       if k[6:] in model_dict.keys()}


    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


    # gpus = [0]
    #将pth转化为pt文件

    #model_jtr = model.to('cuda')
    model_jtr = model.to('cpu')
    model_jtr.eval()
    #将pth转化为pt
    example = torch.rand(1, 3, 480, 640)
    traced_script_module = torch.jit.trace(model_jtr, example)
    traced_script_module.save("/home/yao/Documents/DDRNet.pytorch-main/my/DDRNet23s_imagenet.pt")

    t = 2


if __name__ == '__main__':
    main()
