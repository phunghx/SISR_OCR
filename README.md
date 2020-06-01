# Chương trình nâng cao chất lượng ảnh
## Author: Huỳnh Xuân Phụng
## Date: 5/2020
Hướng dẫn sử dụng

#import library
from sisr_infer import *

import imageio
import  utils

#Tạo model: đường dẫn đến pretrained model trong thư mục checkpoint
SISRModel = SISR()

#dữ liệu đầu vào: ảnh RGB, uint8
input = imageio.imread('./testImg/region2.jpg')

#Dữ liệu đầu ra: ảnh Gray, uint8
output = SISRModel.process(input)

#Lưu file
utils.imsave(output,'testImg2.jpg')