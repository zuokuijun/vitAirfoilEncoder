"""
多输出下的测试：
6层每层100个
10层每层100个
20层每层100个


经过测试10层网络损失最小
下面测试神经元个数对模型的影响：
10层每层60个神经元
10层每层100个神经元
10层每层180个神经元

"""
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model1 = nn.Sequential(
        # 第一层全连接
        nn.Linear(15, 360),
        # nn.ReLU(),
        # nn.Tanh(),
        nn.ELU(),
        # 第二层全连接
        nn.Linear(360, 360),
        # nn.ReLU(),
        # nn.Tanh(),
        nn.ELU(),
        # 第三层全连接
        nn.Linear(360, 360),
        # nn.ReLU(),
        # nn.Tanh(),
        nn.ELU(),
        # 第四层全连接
        nn.Linear(360, 360),
        # nn.ReLU(),
        # nn.Tanh(),
        nn.ELU(),
            # 第四层全连接
        nn.Linear(360, 360),
        # nn.Tanh(),
        # nn.ReLU(),
        nn.ELU(),
        # 第四层全连接
        nn.Linear(360, 360),
        # nn.Tanh(),
        # nn.ReLU(),
        nn.ELU(),
        # 第四层全连接
        nn.Linear(360, 360),
        # nn.Tanh(),
        # nn.ReLU(),
        nn.ELU(),
        # 第四层全连接
        nn.Linear(360, 360),
        # nn.Tanh(),
        # nn.ReLU(),
        nn.ELU(),
        # 第四层全连接
        nn.Linear(360, 360),
        # nn.Tanh(),
        # nn.ReLU(),
        nn.ELU(),
        nn.Linear(360, 3)
        )

    def forward(self, x):
        output = self.model1(x)
        return output


# if __name__ == '__main__':
#
#     root = "./data/val_new2.txt"
#     mlp = MLP()
#     val = MyData(root)
#     label, target = val[0]
#     out = mlp(label)
#     print(out.shape)
#     print(out)
#     # print(imges)
#     # print(labels.shape)