import torch
import torch.backends.cudnn as cudnn
import torchvision
import caffe
import numpy as np
import cv2


import argparse
import os


#isGarry=True
isGarry=False

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir",default='./data/pytorch',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True


net_caffe = caffe.Net('./2deepNet.prototxt', './2deepNet.caffemodel', caffe.TEST)

net_caffe_cv2 = cv2.dnn.readNetFromCaffe('./2deepNet.prototxt', './2deepNet.caffemodel')





def forward_caffe(net,data):
    for input_name,d in zip(net.inputs,data):
        net.blobs[input_name].data[...] = d
        print("*****************")
        print(d)
    rst=net.forward()
    outputs = []
    print("########i43#################")
    print(rst)
    for output_name in net.outputs:
        #print("########")
        x=rst[output_name]
        sum_x_x=0.0
        for _x in np.nditer(x):
            sum_x_x=sum_x_x+_x*_x
            print("tst")
            print(sum_x_x)
        print("sum_x_x= ")
        print(sum_x_x)
        print("np.linalg.norm(x)= ")
        print(np.linalg.norm(x))

        x = x/(np.linalg.norm(x))
        outputs.append(x)
    return outputs
	
def forward_torch(_net,data):
    output=_net.forward(data)
    if isinstance(output,tuple):
        outputs=[]
        for o in output:
            outputs.append(o.data.cpu().numpy())
    else:
        outputs=[output.data.cpu().numpy()]
    return outputs		


def ImagePreProcess(vm_sImagePath, vm_nResizeWidth, vm_nResizeHeight):
    ori_img = cv2.imread(vm_sImagePath)
    #print(ori_img)
    img = cv2.resize(ori_img, (vm_nResizeWidth, vm_nResizeHeight))
    #print(img)

    #img = img.astype(np.float32)
    img = img.transpose((2, 0, 1)) # swapping the axes to go from H,W,C to C,H,W for raw major
    img[[0, 2], ...] = img[[2, 0], ...] # BGR-->RGB

    # transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    img = img / 255.0
    
    img[0, :, :] = (img[0, :, :] - 0.485) / 0.229  # R
    img[1, :, :] = (img[1, :, :] - 0.456) / 0.224  # G
    img[2, :, :] = (img[2, :, :] - 0.406) / 0.225  # B
    #print(img)
    print(img.shape)
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
imgFile="/home/itemhsu/src/py/deepNet/test/data/pytorch/query_1/0112/0112_c1s1_019001_00.jpg"
img=ImagePreProcess(imgFile, vm_nWidth, vm_nHeight)
#print(img)
rsts_caffe_ImagePreProcess = forward_caffe(net_caffe, img)
print("**************")
print(rsts_caffe_ImagePreProcess)
    #net_caffe_cv2.setInput(blob)
    #detections = net.forward()
cvCaffeImgfile(imgFile)
net_caffe_cv2.setInput(img)
detections = net_caffe_cv2.forward()
print("---------------------")
print(detections)



print('finnish')
