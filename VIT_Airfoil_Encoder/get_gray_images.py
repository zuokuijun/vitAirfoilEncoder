# -*- coding: utf-8 -*-
# @Time    : 2022/1/20 10:03 下午
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com
"""
将原始的彩色RGB图片转换为灰度二值化图片
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import  PIL.ImageOps as ops
# import  cv2 as cv

# 首先将图像转换为灰度图像，其次将图像像素进行翻转（因为白色像素数值为255，黑色为0）
# 翻转之后的像素数值为翼型周围的像素数值为0，翼型数值为255
# 最后需要将像素数值统一为0~1之间

# 单个图像灰度转换函数
def convertGray():
    dir = "m6.png"
    img = Image.open(dir)
    # 将图像转换为灰度图
    img = img.convert('L')
    # 将图像像素进行翻转
    img = ops.invert(img)

    w, h = img.size
    # nps = np.array(img)
    # print(nps)
    img.save('m6_gray.png')
    img.show()
    # # 将图像对应的数值写入到相应的文件
    # with open("test.txt", 'w') as f:
    #     for i in range(h):
    #         for j in range(w):
    #             print("----------i-------", i)
    #             print("----------j--------", j)
    #             temp = img.getpixel((j, i))
    #             f.write(str(temp/255)+'  ')
    #             if j == w-1:
    #                 f.write("\n")
    #
    # f.close()

# 多个图像灰度转换函数
def convertGrayx(dir_path):

     # 获取给定目录下的所有文件
     files = os.listdir(dir_path)
     for i in range(len(files)):
         # 获取每一张图片的绝对路径
         print(files[i])
         file  = os.path.join(dir_path, files[i])
         img = Image.open(file)
         # 将图像转换为灰度图
         img = img.convert('L')
         # 将图像像素进行翻转
         img = ops.invert(img)
         img.save("./data/convert_images/{}".format(files[i]))


if __name__ == '__main__':
    dir = "data/images/"
    convertGrayx(dir)

