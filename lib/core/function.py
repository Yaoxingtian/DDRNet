import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from PIL import Image

import torch,sys
import torch.nn as nn
from torch.nn import functional as F
GRANDFA = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(GRANDFA)
from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.utils import Map16, Vedio
# from utils.DenseCRF import DenseCRF

import utils.distributed as dist

vedioCap = Vedio('./output/cdOffice.mp4')
map16 = Map16(vedioCap)

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()
    # from torchsummary import summary

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    tt = 0.00
    
    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch

        images = images#.cuda()
        labels = labels.long()#.cuda()

        losses, _, acc = model(images, labels)
        # print(images.shape)
        # print(labels.shape)
        # summary(model, input_size=[[3, 480, 640],[480, 640]], batch_size=1, device="cpu")
        loss = losses.mean()
        acc  = acc.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        ttt = time.time() - tic
        tt += ttt
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())
        ave_acc.update(acc.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average())
            logging.info(msg)
    print('epoch time :',tt)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer.add_scalar('train_acc', ave_acc.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):       #验证
    model.eval()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            losses, pred, acc = model(image, label)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 100 == 0:
                print(idx)

            loss = losses.mean()
            acc = acc.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())
            ave_acc.update(acc.item())
            print('valid_loss',ave_loss.average(),'ave_acc',ave_acc.average())
    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_acc', ave_acc.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array

#测试
def testval(config, test_dataset, testloader, model,sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            start_time = time.perf_counter()
            image, label, _, name, *border_padding = batch
            # image, label = batch

            # print('name, *border_padding0:',batch)
            # exit()
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)
            elapsed_time = time.perf_counter() - start_time
            elapsed_time = elapsed_time * 1000
            # print(elapsed_time, 'ms')
            #测试数据集，多尺度推理
            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            
            # # crf used for post-processing
            # postprocessor = DenseCRF(   )
            # # image
            # mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225]
            # timage = image.squeeze(0)
            # timage = timage.numpy().copy().transpose((1,2,0))
            # timage *= std
            # timage += mean
            # timage *= 255.0
            # timage = timage.astype(np.uint8)
            # # pred
            # tprob = torch.softmax(pred, dim=1)[0].cpu().numpy()
            # pred = postprocessor(np.array(timage, dtype=np.uint8), tprob)    
            # pred = torch.from_numpy(pred).unsqueeze(0)
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            #得到混淆矩阵
            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred2(image, pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                # print('mean_IoU',mean_IoU)
                # logging.info('mIoU: %.4f' % (mean_IoU))


    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            # print(batch)
            # exit(00)
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                # mean=[0.485, 0.456, 0.406],
                #  std=[0.229, 0.224, 0.225]
                image = image.squeeze(0)
                image = image.numpy().transpose((1,2,0))
                image *= [0.229, 0.224, 0.225]
                image += [0.485, 0.456, 0.406]
                image *= 255.0
                image = image.astype(np.uint8)

                _, pred = torch.max(pred, dim=1)
                pred = pred.squeeze(0).cpu().numpy()
                map16.visualize_result(image, pred, sv_dir, name[0]+'.jpg')
                # sv_path = os.path.join(sv_dir, 'test_results')
                # if not os.path.exists(sv_path):
                #     os.mkdir(sv_path)
                # test_dataset.save_pred(image, pred, sv_path, name)
        vedioCap.releaseCap()
