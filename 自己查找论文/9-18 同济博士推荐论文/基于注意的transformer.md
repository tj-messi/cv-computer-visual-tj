#基于注意的transformer
##abstract
提出了一个新的简单的网络架构，__transformer__，完全基于注意力机制，完全摒弃__递归和卷积__

##introduction
提出了Transformer，这是一种避免重复的模型架构，而是__完全依赖于注意机制__来绘制输入和输出之间的全局依赖关系

##background
Transformer是第一个__完全依赖于自关注__来计算其输入和输出表示的转导模型，而__不使用序列对齐rnn或卷积__

##Model Architecture
Transformer模型：提出了一种全新的序列到序列的模型架构，完全摒弃了__循环神经网络（RNN）和卷积神经网络（CNN）__，仅依赖注意力机制来建立输入和输出之间的全局依赖关系。

__编码器-解码器结构__：模型由编码器和解码器两部分组成，均使用堆叠的自注意力层和逐点全连接层。

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726646742765.png)

###编码器，解码器



###注意力机制

多头注意机制，有点点像NN神经网络里面的W，b矩阵
注意函数可以描述为将查询和一组键值对映射到输出，其中查询、键、值和输出都是向量。输出以加权和的形式计算
![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726647492224.png)

####Scaled Dot-Product Attention（缩放的点积注意力）
![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726648454691.png)

这个是注意力函数：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726648612388.png)

####Multi-Head Attention（多头注意）
