import numpy as np
import cv2
import tensorflow as tf
import pytorch_ssim # package from github: https://github.com/Po-Hsun-Su/pytorch-ssim.git
import torch


# Open the images with cv2.imread() function

def iou(img1, img2):
    # resizing the images to same dimensions
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2,(img1.shape[1], img1.shape[0]), 
                                    interpolation = cv2.INTER_LANCZOS4) 
    # transform image into binary scale
    ret,img2 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY) 

    # Calculate intersection: pixels found in both the prediction and label
    inter = np.logical_and(img1, img2)

    # Calculate union: pixels found in either the prediction or label
    union = np.logical_or(img1, img2)
    
    # intersection over union
    iou = np.sum(inter)/np.sum(union)

    return iou


def ssim(img1, img2):
    # resizing the images to same dimensions
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2,(img1.shape[1], img1.shape[0]), 
                                    interpolation = cv2.INTER_LANCZOS4)

    # put array into a format compatible to run the calculations with tensorflow
    img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0)/255.0
    img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0)/255.0

    # ssim calculation (return a tensor)
    ssim=pytorch_ssim.ssim(img1, img2)

    return ssim.item() # to get the value from the tensor

    
