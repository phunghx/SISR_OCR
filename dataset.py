import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as Ff
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb

import skimage 
from skimage.filters import threshold_yen


class Dataset(object):
    def __init__(self, dataset_type, sigma, scale):
        super().__init__()
        
        self.dataset_type = dataset_type
        
        
        
        self.sigma = sigma
        self.scale = scale
    def len(self):
        1
    def processdata(self, image):
        item = self.load_item_test(image)
        return item

    def load_name(self, index):
        name = self.lr_data_path[index]
        return os.path.basename(name)
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        
    def load_item_test(self, image):
        self.lr_img = image
        if len(self.lr_img.shape)==3 and self.lr_img.shape[0]!=1:
            self.lr_img = self.rgb2gray(self.lr_img)
        elif lr_img.shape[0]==1:
            self.lr_img = self.lr_img[0]
            
        imgh, imgw = self.lr_img.shape[0:2]       
        hr_img = self.lr_img.copy()
        hr_edge = self.load_edge(hr_img,False)
        lr_edge = self.load_edge(self.lr_img,True)
        
        
        return self.to_tensor(self.lr_img), self.to_tensor(hr_img), self.to_tensor(lr_edge), self.to_tensor(hr_edge)
            
    def load_edge(self, img, low_rate=False):         
        canny_img = canny(img.astype(np.float), sigma=self.sigma)
        return canny_img.astype(np.float)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = Ff.to_tensor(img).float()
        
        return img_t

    def resize(self, img, height, width):
        imgh, imgw = img.shape[0:2]

        if imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item