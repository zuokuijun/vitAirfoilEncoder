import os
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional
from torch import nn
from PIL import Image
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import transforms
from data import MyData
from torch.utils.data import DataLoader
# helpers
from torch.utils.tensorboard import SummaryWriter
import  sys
sys.path.append(os.getcwd())

def visualize_grid_attention_v2(img_path, save_path, attention_mask, ratio=1, cmap="jet", save_image=False,
                                save_original_image=False, quality=200):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    print("load image from: ", img_path)
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    print("原始图像类型")
    print(type(img))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    # print(attention_mask.shape)
    # mask = torchvision.transforms.functional.to_pil_image(attention_mask)
    # print(attention_mask.shape)
    # mask = torch.reshape(attention_mask, (img_h, img_w))
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

    if save_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
        img_with_attention_save_path = os.path.join(save_path, img_name)

        # pre-process and save image
        print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path, dpi=quality)

    if save_original_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # save original image file
        print("save original image at the same time")
        img_name = img_path.split('/')[-1].split('.')[0] + "_original.jpg"
        original_image_save_path = os.path.join(save_path, img_name)
        img.save(original_image_save_path, quality=quality)



def pair(t):
    return t if isinstance(t, tuple) else (t, t)  # 主要用于判断输入变量的类型，若输入为元组则直接返回输入值，否则返回一个元组

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        # 8个头。每个头的维度是64, 则总的维度为8 * 64 = 512 , 因为对于单个patch而言，其展开后的维度为512
        # 此时若为8个头， 则分配到每个头的向量维度为64
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # print("input transformer shape")
        # print(x.shape)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # print("get three head dim")
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        # print("print attention")
        # print(attn.shape)
        # print(attn[:, 0:, :])
        # print(attn[:, 7:, :].shape)
        # attn_vis = torch.reshape(attn[:, 7:, :], (1, 65, 65))
        # print(attn_vis.shape)
        # visualize_grid_attention_v2(img_path="../data/val_images/aquilasm.png", save_path= "../data/attention_path", attention_mask=attn_vis)
        # visualize_grid_attention_v2()
        # plt.imshow()
        # plt.plot(attn_vis)
        # writer = SummaryWriter("logs")
        # writer.add_image("show-image1", attn_vis, 1, dataformats="CHW")
        # writer.close()
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Transformer(nn.Module):
    # dim: 单个patch大小 512
    # depth： transformer block 的个数 1
    # heads:  transformer 头的个数 8
    # dim_head：   64
    # mlp_dim： transformer 隐藏层MLP的维度 1024
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):   # N 个transformer block
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x    # Resnet Layer，将原始输入和经过Attention计算得到的矩阵进行相加
            x = ff(x) + x      # Resnet Layer，将MLP的输出和原始输入想加
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)  # 获取图像的宽度和高度
        patch_height, patch_width = pair(patch_size)  # 获取patch的宽度和高度
        # 利用断言抛出异常， 这里表示的是图像大小对patch size 一定是整除的，不然就会抛出异常, 也就是若要对图像做patch，则图像一定要是均分的
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # 图像分割后的patch个数   256/32 * 256/32  =  8 * 8 一共是64个patch
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # 一个patch的维度
        patch_dim = channels * patch_height * patch_width   # 1024  32*32*1
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(

            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),    # 1024 　>> 512
        )
        # 位置编码
        # nn.parameter() 用于将一个固定的tensor()转换为可以训练的数值，使其跟着网络不断优化最终达到最优数值
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # Transformer结果输出
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        # 原始Transformer编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        # 占位网络层
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 10),
            nn.ReLU(),
            nn.Linear(10, num_classes)
        )

    def forward(self, img):
        # 将输入的图像进行序列化
        # print(img)
        x = self.to_patch_embedding(img)  # (1, 64, 1024) > (1, 64, 512) batchsize 为1, 64个patch 每个的维度为512
        # print(x)d
        b, n, _ = x.shape  # 获取原始输入图片的batchsize以及patch的个数
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)  # 定义网络的训练输出
        # 定义单独输出的分类识别的向量
        # print(cls_tokens)
        # print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)  # 将输入的训练数值与定义的token进行拼接
        # 将每个向量的位置编码和patch之后的每个向量相加
        # x += self.pos_embedding[:, :(n + 1)]
        x += self.pos_embedding
        x = self.dropout(x)
        # 输入到Transformer 的Encoder中进行特征提取
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == '__main__':
    root = "../data/label_data/val.txt"
    trans = transforms.Compose([
        # 将图像数据转换为tensor，并且将像素归一化为0~1
        transforms.ToTensor(),
        # 将图片大小统一为216*216
        transforms.Resize((256, 256)),
        #
        # transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    # writer = SummaryWriter("log")
    train_data = MyData(root, transform=trans)
    # print(train_data[0])
    imgs, label = train_data[0]

    # writer = SummaryWriter("logs")
    # writer.add_image("show-image1", imgs, 1, dataformats="HWC")
    # writer.close()
    imgs = imgs.unsqueeze(0)  # 输入图像，batch size=1 , channels=1, width=256, height=256
    vit = ViT(image_size=256,  # 输入的图片大小
            patch_size=32,     # patch 大小
            num_classes=30,    # 最终预测的输出结果
            dim=512,           # 单个patch的维度
            depth=1,           # Transformer block
            heads=8,           # 每个Transformer block 有几个头
            mlp_dim=256)      # Transformer block 中MLP隐藏层的维度
    result = vit(imgs)
    # print(result)
    # print(result.shape)


