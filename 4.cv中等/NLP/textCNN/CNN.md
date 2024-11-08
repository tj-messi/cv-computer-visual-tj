#CNN

##简介

**卷积神经网络**（Convolutional Neural Networks, CNN）是一类包含卷积计算且具有深度结构的前馈神经网络（Feedforward Neural Networks），是深度学习（deep learning）的代表算法之一。

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241108143612.png)

上图中CNN要做的事情是：给定一张图片，是车还是马未知，是什么车也未知，现在需要模型判断这张图片里具体是一个什么东西，总之输出一个结果：如果是车 那是什么车。

最左边是数据输入层(input layer)，对数据做一些处理，比如去均值（把输入数据各个维度都中心化为0，避免数据过多偏差，影响训练效果）、归一化（把所有的数据都归一到同样的范围）、PCA/白化等等。CNN只对训练集做“去均值”这一步。

CONV：**卷积计算层**(conv layer)，线性乘积求和。

RELU：**激励层**(activation layer)，下文有提到：ReLU是激活函数的一种。

POOL：**池化层**(pooling layer)，简言之，即取区域平均或最大。

FC：**全连接层**(FC layer)。

这几个部分中，卷积计算层是CNN的核心。