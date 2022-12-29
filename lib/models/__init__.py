# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys ,os
dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0,dir)

sys.path.append('/home/lyn/Documents/workspace/DDRNet.pytorch-main/lib')


#train

import models.seg_hrnet
import models.seg_hrnet_ocr
import models.ddrnet_23_slim
import models.ddrnet_23
import models.ddrnet_39