# -*- coding: utf-8 -*-
# @Time    : 2022/3/16 5:33 下午
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com

"""
流场预测数据库加载类

前16个表示翼型参数化得到的参数，其次是雷诺数Re、攻角AOA， 后面的五个数据：X Y  U P V
"""

from torch.utils.tensorboard import SummaryWriter
import torch
from PIL import Image
from torch.utils.data import Dataset
torch.set_default_tensor_type(torch.FloatTensor)
import numpy as np
from torchvision import transforms



#**********************DataSet用户自定义数据加载类**********************
class MyData(Dataset):
    def __init__(self, fileDir):
        super(MyData, self).__init__()  # 对继承父类的属性进行初始化
        result = []  # 将所有的数据包装成为Tensor的数据格式，用于后续深度学习的训练
        with open(fileDir, 'r') as f:  # 读取训练数据
            for line in f.readlines():
                line = line.strip('\n')  # 逐行读取，首先将当前行的换行符去掉
                words = line.split()  # 根据空格对字符串进行分割
                word1 = words[0:15]  # 0~14  前15个数据是输入的数据
                input_label = np.array([])
                for i in range(len(word1)):
                    temp1 = float(word1[i])
                    input_label = np.append(input_label, temp1)
                input_label = torch.from_numpy(input_label).float()
                word2 = words[15:]   # 15~17 后三个数据是待预测的数据
                prediction_label = np.array([])
                for j in range(len(word2)):
                    temp2 = float(word2[j])
                    prediction_label = np.append(prediction_label, temp2)
                prediction_label = torch.from_numpy(prediction_label).float()
                result.append((input_label, prediction_label))
        self.result = result

    def __getitem__(self, item):
        input, output = self.result[item]
        return input, output

    def __len__(self):
        return len(self.result)

#**********************DataSet用户自定义数据加载类**********************


if __name__ == '__main__':
    val = MyData("../data_results/val.txt")
    label, target = val[0]
    print(label)
    print(target)


