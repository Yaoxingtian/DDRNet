# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from config import config


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        pred = F.softmax(score, dim=1)

        if pred.numel() == 0:
            cc = torch.unique(target)
            
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        #******** mask 
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        """
        t = tmp_target.cpu()
        t = np.unique(t)
        tt = [0,1,2,3,4,5,255]
        tt = np.array(tt)

        if set(t).issubset(set(tt)):
            t = 2
        else:
            tc = 3
        """

        #tmp_target = tmp_target.cuda()
        # print('self.ignore_label',self.ignore_label)
        # print('tmp_target')
        tmp_target[tmp_target >= self.ignore_label] = 0
        # print('tmp_target',tmp_target)
        #[c, h,w] = tmp_target.size()
        #print(c,h,w)
        # exit()
        tmp_target[tmp_target < 0] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        if pred.numel() == 0:
            cc2 = torch.unique(target)

        if pred.numel()==0:
            min_value = self.min_kept
        else:
            min_value = pred[min(self.min_kept, pred.numel() - 1)]


        #min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
            (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])
