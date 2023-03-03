# -*- coding: utf-8 -*-
# @Time    : 2022/3/16 18:00 下午
# @Author  : zuokuijun
# @Email   : zuokuijun13@163.com

import torch
from torch import nn
from mlp import MLP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mlp_data import MyData

# 需要将输入特征全部归一化到0~1之间
################################################################

batchSize = 1024           # 每次加载的数据量
epochs = 120               # 数据变量次数
learningRate = 0.00005     # 初始化学习率
train_step = 0
test_step = 0
result_model = float("inf")  ##  定义训练结果模型是否保存的标志变量

################################################################
train_path ="../extracted_tecplot_data/train_data_cp/train_files.txt"
val_path = "../extracted_tecplot_data/val_data_cp/val_files.txt"
train_data = MyData(train_path)
test_data = MyData(val_path)
train_loader = DataLoader(dataset=train_data, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batchSize, shuffle=True)
#************************定义新的数据加载方式**************************

mlp = MLP().cuda()
loss = nn.MSELoss().cuda()
writer = SummaryWriter("./mlp_logs/")
optim = torch.optim.Adam(mlp.parameters(), lr=learningRate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1, last_epoch=-1)
for epoch in range(epochs):
    print("---------------------开始第{}轮网络训练-----------------".format(epoch + 1))
    # 训练神经网络
    train_loss = 0
    train_number = 0
    for data in train_loader:
        input, target = data
        input = input.cuda()
        target = target.cuda()
        output = mlp(input)
        target_loss = loss(output, target)
        optim.zero_grad()
        target_loss.backward()
        optim.step()
        train_step = train_step + 1
        train_number += 1  # 统计总计有多少个batchsize
        train_loss += target_loss.item()  # 统计所有batchsize 的总体损失
        if train_step % 100 == 99:
            print("神经网络训练次数[{}], Loss:{}".format(epoch+1, target_loss.item()))
        # writer.add_scalar("train_loss", target_loss.item(), global_step=train_step)
    # writer.add_scalar("epoch_train_loss", train_loss, epoch + 1)  # 当前批次的训练总体损失
    # writer.add_scalar("epoch_train_number", train_number, epoch + 1)  # 当前批次的训练总数
    writer.add_scalar("train_avg_loss", train_loss / train_number, epoch + 1)  # 当前批次训练的平均损失
    lr_scheduler.step()
    # 测试神经网络
    test_total_loss = 0
    test_number = 0
    with torch.no_grad():
        for data in test_loader:
            input, target = data
            input = input.cuda()
            target = target.cuda()
            output = mlp(input)
            test_loss = loss(output, target)
            test_number += 1
            test_total_loss += test_loss.item()
            print("交叉验证集的Losss:{}".format(test_loss.item()))
            # accuracy = (output == labels.squeeze(1)).sum()
            # accuracy = (output.argmax(1) == labels).sum()
            # test_accuracy = test_accuracy + accuracy

    print("测试次数:{},测试数据集上的Loss:{}".format(epoch + 1, test_total_loss/test_number))
    # writer.add_scalar("test_number", test_number, epoch + 1)
    # writer.add_scalar("test_loss", test_total_loss, epoch + 1)
    writer.add_scalar("test_avg_loss", test_total_loss / test_number, epoch + 1)
    if test_total_loss < result_model:   ### 始终保存最小的损失函数值
            result_model = test_total_loss
            state = {'models': mlp.state_dict(),      ### 模型保存用于神经网络模型的加载与续算
                     'optimizer': optim.state_dict(),
                     'epoch': epoch + 1}
            torch.save(state, "./models/mlp_{}.pth".format(epoch + 1))
            print("模型已保存！")


