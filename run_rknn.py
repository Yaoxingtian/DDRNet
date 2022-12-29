from rknn.api import RKNN
import time
import numpy as np
import torch
import cv2,glob
from

colors =[
                [255, 191, 0],  # 深天蓝
                [138, 223, 152],  # 地板： 草地绿
                [255, 248, 248],  # 墙壁：幽灵白
                [0, 165, 255],  # 椅子：橙子色
                [213, 176, 197],  # 家具：淡紫色（薰衣草色）
                [0, 255, 0],  # 其他物体：	亮绿色
                [138, 223, 152],  # 窗口：草地绿
                [0, 255, 255],  # 床：黄色（颜色与颜色表的RB值位置相反）
                [156, 158, 222],  # 窗口：薰衣草紫
                [255, 0, 0],  # 纯蓝色
                [0, 255, 0],  # 线缆：亮绿色
                [79, 79, 47],  # 深石板灰
        ]
label_colours = dict(zip(range(5), colors))
def decode_segmap(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 5):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r     / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb
def transformY(image,img_size):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    image = torch.from_numpy(image).float()
    return image

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= self.mean
    image /= self.std
    return image

RKNN_MODEL = '/home/yao/Documents/DDRNet.pytorch-main/tools/output/ddrnet23_3566.rknn'
rknn = RKNN(verbose=True)

model = rknn.load_rknn(RKNN_MODEL)
ret = rknn.init_runtime(target='rk3566')
if ret != 0:
    print('Init runtime environment failed')
    exit(ret)
print('loading model done')



paths = glob.glob(os.path.join(WSI_MASK_PATH, '*'))
# bb = Cityscapes(config.DATASET.ROOT,config.DATASET.TEST_SET)
# cap = cv2.VideoCapture('G:/googleDownload/20200916/WIN_20210203_14_29_38_Pro.mp4')
# cot = 1
for path in paths:
    frame = cv2.imread(path)
    # print(cot)
    # cot = cot+1
    # if cot<17000:
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
    print(path)
    image2 = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    img = image2
    frame = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    images1 = np.array(frame, dtype=np.uint8)
    img_size = (480, 640)
    images1 = transformY(images1, img_size)

    out = None
    if x > 0:
        numel = sum([x.numel() for x in images1])
        storage = images1.storage()._new_shared(numel)
        out = images1.new(storage)
    x = x - 1
    yz = (images1,)
    images = torch.stack(yz, 0, out=out)
    images = images.to("cuda")

    image2 = input_transform(image2)
    image2 = image2.transpose((2, 0, 1))
    image2 = image2[np.newaxis, :]

    image2 = torch.from_numpy(image2)
    # 将pth转化为pt
    # image3 = image2.to('cuda')
    # example = torch.rand(1, 3, 480, 640)
    # traced_script_module = torch.jit.trace(model_jtr, example)
    # traced_script_module.save("ddrnet23_slim.pt")
    start = time.time()
    outputs = rknn.inference(inputs=[image2])
    end = time.time()
    print('inference time: ', end - start)
    print('done')

    # start = timeit.default_timer()
    pred = test_dataset.multi_scale_inference(config, model, image2)
    # end = timeit.default_timer() - start
    # end = end * 1000
    # print(end, 'ms')
    pred2 = np.squeeze(pred.data.max(1)[1].cpu().numpy(), axis=0)
    decoded = test_function.decode_segmap(pred2)
    img_input = np.squeeze(images.cpu().numpy(), axis=0)

    blend = img_input * 0.1 + decoded * 0.9

    im_name = path.split('/')[-1]
    cv2.imwrite('{}/{}'.format(save_path, im_name), pred2)
    # print(img)
    # exit()
    # cv2.imshow('Img_DDRNet',img)
    # cv2.imshow('DDRNet', blend)
    # cv2.waitKey(1)