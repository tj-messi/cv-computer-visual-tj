#End-to-End Object Detection with Transformers

##abstract
在其他方式进行目标检测的时候会在最后一步使用到NMS来取出冗余的框，这会导致模型调参不好调，部署困难。

而__transformer__实现了__端到端的全局建模__，直接呈现了一个__集合的预测__问题

我们的方法简化了检测管道，__有效地消除了许多手工设计的组件__，如__非最大抑制过程__ 或 __锚生成__，这些组件显式地编码了我们对任务的先验知识。

##1.introduction
现有的预测期都是用一个人工的调参控制的内容来实现__集合预测(边框预测)__

全局的特征能很好的减少多个类似的框

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1727192867973.png)

1.利用传统的__CNN__网络将输入图像转换为特征图

2.__transformer__提取学习全局特征

3.__transformer__生成多个预测框

4.根据__预测框__和__ground truth__框匹配计算__loss损失__

detr善于检测大物体，暂时对小物体有点障碍

detr训练时间很久，但是泛用性很广

##2.related work

###2.1objection detection
大多数现代目标检测方法都是根据最初的猜测做出预测的，__两阶段的检测器__基于的是__之前的预测方框建议__

##3.Detr模型

模型示意图：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1727224046725.png)

###3.1 预测损失

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1727223323202.png)

每一次都会输出N（自己设定）的预测框个数，然后进行__二分图匹配__,自己可以设置好一个__L-1 loss函数__ 和 __IOU损失函数__的结合

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1727224436393.png)

###3.2 模型架构

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1727225002905.png)

1.输入图像: 3-800-1066

2.__CNN__卷积抽取特征：2048(通道数量)--25(800/32)--34(1066/32)

3.位置编码：256--25--34

4.展平特征传入transformer：850--256

5.transformer 解码器 ： 100--256

6.__FFN__输出预测框

7.二分图匹配检测损失回传loss来更新模型