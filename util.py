import os
import glob
import random
from pathlib import Path

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import cv2
import png
from PIL import Image
import flow_vis

import numpy as np
import matplotlib.pyplot as plt

def read_png_flow(flow_file):
    """
    Read from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2**15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow


def write_png_flow(F_uv, filename):
    m,n,k = F_uv.shape
    I = np.zeros((m, n, 3))
    if k == 2:
        F_val = np.ones((m, n))
    elif k ==3:
        F_val = F_uv[:,:,2]
    
    I[:,:,0] = np.maximum(np.minimum(F_uv[:,:,0]*64+2**15, 2**16-1), 0)
    I[:,:,1] = np.maximum(np.minimum(F_uv[:,:,1]*64+2**15, 2**16-1), 0)
    I[:,:,2] = np.maximum(np.minimum(F_val, 1), 0)

    cv2.imwrite(filename, cv2.cvtColor(I.astype(dtype='uint16'), cv2.COLOR_BGR2RGB))
    return 0

def resample_flow(flow, size):
    """
    flow: flow map to be resampled
    size: new flow map size. Must be [height,weight]
    """
    original_image_size = flow.shape
    in_height = flow.shape[0]
    in_width = flow.shape[1]
    out_height = size[0]
    out_width = size[1]
    out_flow = np.zeros((out_height, out_width, 2))
    # find scale
    height_scale =  float(in_height) / float(out_height)
    width_scale =  float(in_width) / float(out_width)

    [x,y] = np.meshgrid(range(out_width), range(out_height))
    xx = x * width_scale
    yy = y * height_scale
    x0 = np.floor(xx).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(yy).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0,0,in_width-1)
    x1 = np.clip(x1,0,in_width-1)
    y0 = np.clip(y0,0,in_height-1)
    y1 = np.clip(y1,0,in_height-1)

    Ia = flow[y0,x0,:]
    Ib = flow[y1,x0,:]
    Ic = flow[y0,x1,:]
    Id = flow[y1,x1,:]

    wa = (y1-yy) * (x1-xx)
    wb = (yy-y0) * (x1-xx)
    wc = (y1-yy) * (xx-x0)
    wd = (yy-y0) * (xx-x0)
    out_flow[:,:,0] = (Ia[:,:,0]*wa + Ib[:,:,0]*wb + Ic[:,:,0]*wc + Id[:,:,0]*wd) * out_width / in_width
    out_flow[:,:,1] = (Ia[:,:,1]*wa + Ib[:,:,1]*wb + Ic[:,:,1]*wc + Id[:,:,1]*wd) * out_height / in_height

    return out_flow

def flow2rgb(flow):
    flow = flow.transpose((1, 2, 0))
    return flow_vis.flow_to_color(flow, convert_to_bgr=False)

def predict_flow(model, im1, im2):
    flow2 = model(im1, im2)[0]
    flow1 = 20 * F.interpolate(flow2, size=(im1.shape[2], im1.shape[3]), mode='bilinear', align_corners=False)

    return flow1
