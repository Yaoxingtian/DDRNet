# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,sys
sys.path.append('/home/lyn/Documents/workspace/DDRNet.pytorch-main/lib')
from config import config
# from .data import GZdata_add_ikea0124 as GZdata_add_ikea0124
from .data import cityscapes as cityscapes
from .data import sunRGBD_8945 as sunRGBD_8945

from .lip import LIP as lip
from .pascal_ctx import PASCALContext as pascal_ctx
from .ade20k import ADE20K as ade20k
from .map import MAP as map
from .cocostuff import COCOStuff as cocostuff