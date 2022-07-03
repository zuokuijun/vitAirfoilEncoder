#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/7/3 下午3:38
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com
# @File    : plot_airfoil.py
# @Software: PyCharm
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import  PIL.ImageOps as ops

class airfoil():

    ###EN: plot uiuc single airfoil
    ###CN: 绘制UIUC数据库中的单个机翼图像
    def plot_single_airfoil(self):
        x = []
        y = []
        with open("./data/uiuc/rae2822.dat", "r") as f:
            for line in f.readlines():
                line = line.strip("\n")
                line = line.split()
                x.append(float(line[0]))
                y.append(float(line[1]))
        fig, ax = plt.subplots(1, 1, figsize=[6, 3])
        # 定义画图的数据、类别等
        ax.plot(x, y, color='k', lw=3)
        # 设置坐标的显示范围
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.2, 0.2])
        # 不显示坐标
        plt.axis('off')
        plt.show()

    ## 绘制多个机翼的图像并且保存
    def plot_multi_airfoil_save(self):
        dir = "data/uiuc/"   # 机翼坐标数据库文件所在的路径
        files = os.listdir(dir)
        print("UIUC机翼数据库绘制，总个数为{}".format(len(files)))
        for i in range(len(files)):
            print("绘制第{}个机翼,{}".format(i, files[i]))
            name = files[i].split(".")[0]   # 获取机翼名称
            path = os.path.join(dir, files[i])  # 获取每个机翼的绝对路径
            x = []
            y = []
            with open(path, "rt") as f:
                for line in f.readlines():
                    # print(line)
                    if line.strip() == "":  # 若当前行为空行，则直接跳过
                        continue
                    line = line.strip("\n")
                    lines = line.split()
                    # print(lines)
                    x.append(float(lines[0]))
                    y.append(float(lines[1]))

            fig, ax = plt.subplots(1, 1, figsize=[6, 3])
            # 定义画图的数据、类别等
            ax.plot(x, y, color='k', lw=3)
            # 设置坐标的显示范围
            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-0.2, 0.2])
            # 不显示坐标
            plt.axis('off')
            plt.savefig("./data/images/{}.png".format(name))
            x.clear()
            y.clear()
            plt.close()
            f.close()



if __name__ == '__main__':
    wing = airfoil()
    # wing.plot_single_airfoil()
    wing.plot_multi_airfoil_save()