---
layout: post
title: "EfficientNet"
date: 2020-04-11
excerpt: "Recognition. Rethinking Model Scaling for Convolutional Neural Networks"
tags: [recognition, EfficientNet, NAS]
comments: true
---

paper: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
(Zhang](https://arxiv.org/abs/1905.11946)

code: [官方tensorflow](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

[PyTorch版本](https://github.com/lukemelas/EfficientNet-PyTorch)

该团队又出了一篇目标检测的EfficientDet，可以看作是将EfficientNet迁移到目标检测方向，同时做了改进
paper: [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)


### Abstact
1. 在有限资源下对ConvNet进行scale up一般都是能够对性能带来提升的，一般的scale up的方式有增大channel width、网络going deeper、输入增大resolution几种方式，但是现阶段的很多方式都是对单一的某一种方式进行实验，且基本上都需要手动干预进行裁剪等，缺乏一种系统的scale up的方式。
2. 基于上述的情况，作者提出一种compound scale的方法，主要就是不需要过多的人工干预，实现channel width、net depth、resolution之间的balance。性能飞起，如下：
    * ![EfficientNet model size VS accuracy](./cls_attachments/cls3_EfficientNet_model_size_accuracy.png)
3. 整体来看，思路不难理解，但是对性能的优化有不少细节，比如用NAS去搭建baseline网络，用mobilenet的mobile inverted bottleneck作为NAS的building block，引入SENet的SE模块（squeeze-and-excitation）等
    * 同时，算力的问题，官方公布的是用TPU计算。没有实际验证，不知道对算力的要求如何；


### Details
1. compound model scaling的目标方程，如下：
    * ![EfficientNet target](./cls_attachments/cls3_EfficientNet_target.png)
    * 即在给定的任意资源下，获得最好的模型精度

2. 作者文中具体分析了不同方式的scaleup（width、depth、resolution），最终证明是需要保持三者之间的balance，目标如下：
    * ![EfficientNet compound scaling](./cls_attachments/cls3_EfficientNet_compound_scaling.png)
    * ![EfficientNet different scaling](./cls_attachments/cls3_EfficientNet_model_scaleup.png)
    * 其中，width、depth、resolution的三个对应的参数alpha、beta、gamma通过grid search的方式比较容易确定

3. 上述的scale up都是基于某一个给定的模型的，如作者文中所述的一样，需要先保证能有一个好的baseline，即EfficientNet，如何做？
    * 这里用的NAS的方式（neural architecture search）设计得到EfficientNet-B0

4. 如何在EfficientNet-B0的基础上scale up？
    * 首先固定alpha、beta、gamma的幂pha，然后用grid search的方式，确定alpha、beta、gamma；所谓grid search其实就是列举一些候选的参数，组合进行对比实验
    * 固定上一步得到的alpha、beta、gamma，用不同的pha对网络scale up得到不同的网络结构，即EfficientNet-B1到B7