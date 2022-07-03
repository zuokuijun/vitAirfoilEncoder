#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/7/3 下午4:54
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com
# @File    : get_airfoil_map.py
# @Software: PyCharm
import os
import matplotlib.pyplot as plt
from PIL import Image


# 获取翼型图像三通道的热力图
def get_airfoil_map(file_path):
    files = os.listdir(file_path)
    print("files size is {}".format(len(files)))
    for i in range(len(files)):
        print("The precessing is {}".format(i))
        file = os.path.join(file_path, files[i])  # 获取机翼图像的路径
        img = Image.open(file)
        plt.imshow(img)
        plt.axis("off")
        plt.margins(0, 0)
        ## 将机翼图像按照热力图的形式进行保存
        plt.savefig("./data/airfoil_map/{}".format(files[i]),  bbox_inches='tight', pad_inches=0.0)
        plt.close()
        # plt.show()
        # #
        # ## 将保存的机翼图像转换为RGB 3通道格式
        RGB_mode = Image.open("./data/airfoil_map/{}".format(files[i]))
        RGB_mode = RGB_mode.convert('RGB')
        RGB_mode.save("./data/airfoil_map/{}".format(files[i]))



if __name__ == '__main__':
    dir = "data/convert_images/"
    get_airfoil_map(dir)
