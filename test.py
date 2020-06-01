from sisr_infer import *

import imageio
import  utils

SISRModel = SISR()
input = imageio.imread('./testImg/region2.jpg')

output = SISRModel.process(input)
utils.imsave(output,'testImg2.jpg')
