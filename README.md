# Deep  Attention Network (DAN)

---

The repository mainly records how to use VIT to code the airfoil geometry and how to use the encoded information to reconstruct the flow field of the airfoil. 🖐

More details about this work can be found from our paper:

[Fast aerodynamics prediction of laminar airfoils based on deep attention network](https://pubs.aip.org/aip/pof/article-abstract/35/3/037127/2882158/Fast-aerodynamics-prediction-of-laminar-airfoils?redirectedFrom=fulltext)

---



## 1、Airfoil   Visual   Transformer (VIT)  geometry encoding

---

1、timm_ V2.py is implemented using the timm visual algorithm function library. The traditional implementation method can refer to VIT_author.py file.

pytorch-image-models references：

author_home_page：https://github.com/rwightman

pytorch-image-models open source code：https://github.com/rwightman/pytorch-image-models

Zhi hu：[https://zhuanlan.zhihu.com/p/350837279](https://zhuanlan.zhihu.com/p/350837279)

2、Visual Transformer Code Reference for Attention Visualization：https://github.com/zuokuijun/Transformer-Explainability

3、 VIT_ Airfoil_ Encoder is a Pycharm engineering file that uses Transformer to encode geometric parameters for UIUC airfoil database

### How to use  ?👉👉👉👉👉

* `cd VIT_Airfoil_Encoder`

* run `python plot_airfoil.py` generate  airfoil  images. 
* run `python get_gray_images.py` generate airfoil gray images
* run `python get_airfoil_map.py` generate airfoil three channel  airfoil  heat-map  images
* run `python vit_explain.py`  get airfoil  geometry encoding information

## e1412 airfoil  attention  visualization

<p align="center">
    <img src="./images/e1214.png"  width="300" height="200"/>
       <img src="./images/e1214_attention.png"  width="300" height="200"/>




## naca4412 airfoil  attention  visualization

<p align="center">
    <img src="./images/naca4412.png"  width="300" height="200"/>
    <img src="./images/naca4412_attentions.png"  width="300" height="200"/>
</p>


---

## 2、 Airfoil flow field prediction

* `cd VIT_flow_field_prediction`

* run `train.py` file to train DAN  

* run `mlp_test.py`  to get  DAN prediction  results  

  **Tips**:  The test model and test data can be found in [[vitAirfoilEncoder](https://www.kaggle.com/datasets/kuijunzuo/vitairfoilencoder/data)]([vitAirfoilEncoder](https://www.kaggle.com/datasets/kuijunzuo/vitairfoilencoder/data))



## If you feel that our work is helpful to you, please cite our work in your article



```
@article{zuo2023fast,
  title={Fast aerodynamics prediction of laminar airfoils based on deep attention network},
  author={Zuo, Kuijun and Ye, Zhengyin and Zhang, Weiwei and Yuan, Xianxu and Zhu, Linyang},
  journal={Physics of Fluids},
  volume={35},
  number={3},
  year={2023},
  publisher={AIP Publishing}
}
```









