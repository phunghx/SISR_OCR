"""
chương trình nâng cao chất lượng ảnh 
author: Huỳnh Xuân Phụng
Date: 5/2020
Input: region (numpy array [width, height]) (gray scale 0..1)
Output: enhanced region
"""
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack

import cv2
from models import EdgeModel, SRModel
from config import Config
from dataset import Dataset
class SISR(object):
    def __init__(self):
        self.config =  Config('./config.yml')  
        if torch.cuda.is_available():
           self.config.DEVICE = torch.device("cuda")
           torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
        else:
           self.config.DEVICE = torch.device("cpu")
        self.config.PATH = './checkpoint'
        torch.manual_seed(self.config.SEED)
        torch.cuda.manual_seed_all(self.config.SEED)
        np.random.seed(self.config.SEED)
        random.seed(self.config.SEED)
        
        self.dataset = Dataset('test',sigma=self.config.SIGMA, scale=self.config.SCALE)
        self.sr_model = SRModel(self.config).to(self.config.DEVICE)
        self.sr_model.load()
        self.sr_model.eval()
        
    def process(self,image):
        items = self.dataset.processdata(image)
        lr_images, hr_images, lr_edges, hr_edges = self.cuda(*items)
        outputs = self.sr_model(lr_images.unsqueeze(0), lr_edges.unsqueeze(0))
        output = self.postprocess(outputs)
        return output
    
    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)
    def high_pass(self,im):
        F1 = fftpack.fft2((im).astype(float))
        F2 = fftpack.fftshift(F1)
        (w, h) = im.shape
        half_w, half_h = int(w/2), int(h/2)
        n = 20
        F2[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0
        im1 = fftpack.ifft2(fftpack.ifftshift(F2)).real
        im1 = 255*(im1 - np.min(im1))/np.ptp(im1).astype(int)
        return im1.astype(int)
    def postprocess(self, img):
        # [0, 1] => [0, 255]
        #img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        img = img.cpu().detach().numpy().astype(np.float32)
        im_low = cv2.bilateralFilter(img[0,:,:,0], 9, 150, 150, cv2.BORDER_DEFAULT)
        im_low = im_low*255.0
        img = img * 255.0
        im_low = im_low.astype(int)
        #im_high = self.high_pass(img[0,:,:,0])
        image = self.dataset.lr_img
        image = image.astype(int)
        if im_low.shape[0]!=image.shape[0]:
            temp = np.ones_like(image) * 255
            temp[0:im_low.shape[0],0:im_low.shape[1]] = im_low
            im_low = temp.copy()
            temp[0:img.shape[1],0:img.shape[2]] = img[0,:,:,0]
            im_ori = temp.copy()
        else:
            im_ori = img[0,:,:,0]
        
        result = cv2.normalize((im_ori+im_low),0,255,norm_type=cv2.NORM_MINMAX)
        
        return result
    

