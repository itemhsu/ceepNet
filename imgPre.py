import torch
import torch.backends.cudnn as cudnn
import torchvision
import caffe
import numpy as np
import cv2


import argparse
import os


def ImagePreProcess(vm_sImagePath, vm_nResizeWidth, vm_nResizeHeight):
    ori_img = cv2.imread(vm_sImagePath)
    #print(ori_img)
    img = cv2.resize(ori_img, (vm_nResizeWidth, vm_nResizeHeight))
    #print(img)

    #img = img.astype(np.float32)
    img = img.transpose((2, 0, 1)) # swapping the axes to go from H,W,C to C,H,W for raw major
    print(img)
    img[[0, 2], ...] = img[[2, 0], ...] # BGR-->RGB

    # transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    img = img / 255.0
    
    img[0, :, :] = (img[0, :, :] - 0.485) / 0.229  # R
    img[1, :, :] = (img[1, :, :] - 0.456) / 0.224  # G
    img[2, :, :] = (img[2, :, :] - 0.406) / 0.225  # B
    print(img)
    #print(img.shape)
    swapped = np.moveaxis(img, 1, 1)  # shape (y_pixels, x_pixels, n_bands)
    arr4d = np.expand_dims(swapped, 0)
    print("arr4d.shap=")
    print(arr4d.shape)
    return arr4d
def cvCaffeImgfile(vm_sImagePath):
    ori_img = cv2.imread(vm_sImagePath)
    
    blob = cv2.dnn.blobFromImage(
        cv2.resize(ori_img, (128, 64)), 
        0.225, 
        (128, 64), 
        (104.0,117.0,123.0))
    #print(blob.shape)
    #print(blob)

    # pass the blob through network
    #net.setInput(blob)

vm_nWidth = 64
vm_nHeight = 128
img=ImagePreProcess("img.jpg", vm_nWidth, vm_nHeight)
