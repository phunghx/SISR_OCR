# Chương trình nâng cao chất lượng ảnh
## Author: Huỳnh Xuân Phụng
## Date: 5/2020
<br/>
Yêu cầu: <br/>
pytorch 1.3.1 <br/>
opencv 
<br/>
(Có thể xem chi tiết version trong thư mục cài đặt trên máy chủ FPT 10.3.9.222:/root/miniconda)


Hướng dẫn sử dụng

#import library
from sisr_infer import *

import imageio
import  utils

#Tạo model: đường dẫn đến pretrained model trong thư mục checkpoint <br/>
SISRModel = SISR()
<br/>
<br/>
#dữ liệu đầu vào: ảnh RGB, uint8 <br/>
input = imageio.imread('./testImg/region2.jpg')
<br/>
<br/>
#Dữ liệu đầu ra: ảnh Gray, uint8 <br/>
output = SISRModel.process(input)
<br/>
<br/>
#Lưu file <br/>
utils.imsave(output,'testImg2.jpg')
