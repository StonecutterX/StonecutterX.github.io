---
layout: post
title: "Strip Pooling"
date: 2020-04-11
excerpt: "Scene Parsing, Strip Pooling: Rethinking Spatial Pooling for Scene Parsing."
tags: 
- segmentation
- strip pooling
- spatial pooling
comments: true
---

paper: [Strip Pooling: Rethinking Spatial Pooling for Scene Parsing](http://arxiv.org/abs/2003.13328)

code: [https://github.com/Andrew-Qibin/SPNet](https://github.com/Andrew-Qibin/SPNet.)

### Abstract
1. CVPR2020的一篇文章，通过改进空间池化层来优化场景分割的任务。其出发点是，传统的标准pooling多是方形，而实际场景中会有一些物体是长条形，需要模型能够尽可能捕获一个long-range的dependencies。因此，作者引入来一种*long but narrow kernel*。
2. 本文主要的几个关注点是：
    * 引入*strip pooling*，即前面说的*long but narrow kernel*
    * 在上面的基础上，构造来*strip pooling module(SPM)*，使得该结构在现有的网络结构中能够即插即用
    * 进一步将strip pooling和standard spatial pooling组合，提出*mixed pooling module*，即综合标准的spatial pooling和strip pooling，以兼顾各种shape的物体的分割
    * 基于上面的所有改进，提出来*SPNet*，验证前面几点的有效性

#### Details
1. strip pooling如何实现？
    * 实际物体中有不少物体依赖long-range的dependencies，那么从pooling的角度解决这一的问题，最直接的方式应该就是把pooling操作改成*long but narrow*的形状，即本文说的*strip pooling*
    * 实现上，很简单，相当于把标准的spatial pooling的kernel的宽或高置为1，然后每次取所有水平元素或垂直元素相加求平均（average pooling，按文中公式的表述，每次取的所有水平元素或垂直元素的个数分别为输入的tensor的w和h），即只是改变了spatial pooling中采样的方式，对应论文中的公式2和3
    * 在作者公布的PyTorch源码中，作者是通过AdaptiveAvgPooling实现的
        * PyTorch的AdaptiveAvgPooling操作，是由用户指定输入和输出的size，该操作内部会去计算对应的stride和kernel size，可以[参考这里](https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work)
    * 利用strip pooling和spatial pooling的效果对比可以看下图
        * ![](https://img2020.cnblogs.com/blog/1467786/202004/1467786-20200422002503764-46240669.png)

2. strip pooling module
    * 为了能让strip pooling实现在现有的不同网络结构中即插即用，作者设计了strip pooling module，即将strip pooling封装到该模块内部，保证对于输入特征图，经过SPM模块后，是已经执行过strip pooling操作的特征图
    * 如下图：
        ![](https://img2020.cnblogs.com/blog/1467786/202004/1467786-20200422004936497-503019046.png)
        * 对于一个输入tensor，用两个pathway分别处理水平和垂直的strip pooling，然后再expand到输入tensor的原尺寸（看下文中的代码，这个expand应该是通过上采样的插值实现的*interprolate*）
        * 然后将两个pathway的结果相加进行融合；之后再添加一个`1x1 conv`（改变channel个数），然后加上sigmoid激活函数
        * 同时SPM中有一个类似residual的identity map操作，将上面两个pathway融合后经过sigmoid的结果直接通过element-wise的乘法融合到一起。**这里相当于上一步sigmoid得到的是一个权重矩阵，得到输入tensor中每个位置的特征的重要性，因此，上面2路的pathway其实是可以不用重新训练的，有点像attention机制**

3. mixed pooling module
    * 如果因为上面的考虑将网络中的所有pooling全部换成strip pooling操作，则必然会影响原来的非长条物体的效果，就得不偿失了。因此，作者将strip pooling和pyramid pooling都加入进来，构造成mixed pooling module
    * 其中，strip pooling用于解决long-range dependencies，而轻量级的pyramid pooling用于解决short-range dependencies

4. 实现代码
    * [SPNet部分](https://github.com/Andrew-Qibin/SPNet/blob/master/models/spnet.py)，其backbone是resnet系列
    * 参考源码[StripPooling部分](https://github.com/Andrew-Qibin/SPNet/blob/master/models/customize.py)
        ```Python
        ### 通过AdaptiveAvgPool2d实现strip pooling
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        ## SPM模块
        def forward(self, x):
            _, _, h, w = x.size()
            x1 = self.conv1_1(x)
            x2 = self.conv1_2(x)
            x2_1 = self.conv2_0(x1)
            x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
            x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
            x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
            x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
            x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
            x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
            out = self.conv3(torch.cat([x1, x2], dim=1))
            return F.relu_(x + out)
        ```

#### Conclusions
1. 个人感觉strip pooling这个思路还是比较有道理，且实现也比较简单，对于交通场景中的，路障，车道线这类物体，应该是会有所帮助的。
2. 再者，其中strip pooling这个思路，如果和非对称卷积结合到一起，有没有可能会进一步提升文中阐述的需要long-range dependencies的物体的效果呢？如[ACNet](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf)的做法