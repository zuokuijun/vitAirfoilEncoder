import os

import torch
import numpy as np
from mlp import MLP
import time
"""
model_parameters:  1046523   
"""
def get_prediction_result():
    model = MLP()
    # 模型加载
    checkpoint = torch.load("./models/3090_1_seed_2_97.pth", map_location=torch.device('cpu'))   ### 加载神经网络模型
    model.load_state_dict(checkpoint['models'])
    # print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    # # 将模型预测得到的UPV数据进行保存
    result = open("../Utils/large_data_set/p_cp/RAE2822_900_1_large1_cp_pre2.txt", mode='w', encoding='utf8')  # 将预测得到的流场数据进行保存
    start_time = time.time()
    with open("../Utils/large_data_set/p_cp/RAE2822_900_1_large1_cp_nor.txt", 'r') as f:  # 读取测试数据
          for line in f.readlines():
             line = line.strip('\n')     # 将当前字符串中的回车符去掉
             dataset = line.split()
             input = dataset[0:15]       # 获取前15个字符作为神经网络的输入数据
             input_label = np.array([])  # 定义临时变量将数据转换为float类型 便于后续转为torch类型进行预测任务
             for j in range(len(input)):
                temp = float(input[j])
                input_label = np.append(input_label, temp)
             input_label = torch.from_numpy(input_label).float() # 将数据转换为torch类型
             model.eval()
             with torch.no_grad():

                output = model(input_label)


                output = output.numpy()
                out = ""
                for i in range(len(output)):
                    out += str(output[i]) + " "
                result_string = out + "\n"
                result.writelines(result_string)
                result.flush()
    result.close()
    end_time = time.time()
    print("time={}".format(end_time - start_time))

def  get_multi_prediction_results():
    model = MLP()
    # 模型加载
    checkpoint = torch.load("./models/3090_1_seed_2_97.pth", map_location=torch.device('cpu'))  ### 加载神经网络模型(测试大批量翼型数据的预测时间)
    model.load_state_dict(checkpoint['models'])


    path = "../extracted_tecplot_data/test_data_cp/mlp_cp_test/multi_data_test/"
    files = os.listdir(path)
    start_time = time.time()
    for i in range(len(files)):
         with open(path+files[i], 'r') as f:  # 读取测试数据
            for line in f.readlines():
                line = line.strip('\n')  # 将当前字符串中的回车符去掉
                dataset = line.split()
                input = dataset[0:15]  # 获取前15个字符作为神经网络的输入数据
                input_label = np.array([])  # 定义临时变量将数据转换为float类型 便于后续转为torch类型进行预测任务
                for j in range(len(input)):
                    temp = float(input[j])
                    input_label = np.append(input_label, temp)
                input_label = torch.from_numpy(input_label).float()  # 将数据转换为torch类型
                model.eval()
                with torch.no_grad():
                    output = model(input_label)
    end_time = time.time()
    print("time={}".format(end_time - start_time))
if __name__ == '__main__':
    get_prediction_result()
    # get_multi_prediction_results()