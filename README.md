# VIT_Airfoil_Encoder

这个仓库主要记录了如何利用VIT对输入的机翼图像进行隐式编码，编码的向量用来描述机翼的几何特征

## e1412 机翼视觉Transformer 几何编码

<p align="center">
    <img src="./images/e1214.png"  width="300" height="200"/>
       <img src="./images/e1214_attention.png"  width="300" height="200"/>




## naca4412 机翼视觉Transformer 几何编码

<p align="center">
    <img src="./images/naca4412.png"  width="300" height="200"/>
    <img src="./images/naca4412_attentions.png"  width="300" height="200"/>
</p>


---





1、timm_v2.py 使用timm视觉算法函数库实现，传统的实现方法可以参考VIT_author.py文件里面关于VIT的实现方法

pytorch-image-models参考：

作者主页：https://github.com/rwightman

pytorch-image-models开源代码：https://github.com/rwightman/pytorch-image-models

知乎：[https://zhuanlan.zhihu.com/p/350837279](https://zhuanlan.zhihu.com/p/350837279)

2、视觉Transormer注意力可视化的代码参考：https://github.com/zuokuijun/Transformer-Explainability

3、 VIT_Airfoil_Encoder  为针对UIUC翼型数据库利用Transformer进行几何参数编码的Pycharm工程文件
