#基于注意的transformer
##abstract
提出了一个新的简单的网络架构，__transformer__，完全基于注意力机制，完全摒弃__递归和卷积__

##introduction
提出了Transformer，这是一种避免重复的模型架构，而是__完全依赖于注意机制__来绘制输入和输出之间的全局依赖关系

RNN处理文本的时候需要顺序输入，利用h(t-1)来计算ht。

用__多头注意力机制__模拟卷积神经网络一个__多输出通道__的效果

而transformer是不需要的，可以提高并行性

##background
Transformer是第一个__完全依赖于自关注__来计算其输入和输出表示的转导模型，而__不使用序列对齐rnn或卷积__

##Model Architecture
Transformer模型：提出了一种全新的序列到序列的模型架构，完全摒弃了__循环神经网络（RNN）和卷积神经网络（CNN）__，仅依赖注意力机制来建立输入和输出之间的全局依赖关系。

__编码器-解码器结构__：模型由编码器和解码器两部分组成，均使用堆叠的自注意力层和逐点全连接层。

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726646742765.png)

###编码器，解码器

输入一系列的x1...xn，就比如说是你的文本。然后编码器输出成z1...zn，转化为__机器能理解的向量__

解码器接受z之后生成m个y:y1....ym

###注意力机制

多头注意机制，有点点像NN神经网络里面的W，b矩阵
注意函数可以描述为将查询和一组键值对映射到输出，其中查询、键、值和输出都是向量。输出以加权和的形式计算
![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726647492224.png)

####Scaled Dot-Product Attention（缩放的点积注意力）
![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726648454691.png)

这个是注意力函数：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726648612388.png)

__自注意力层__：在编码器和解码器中均使用了自注意力层，允许模型在处理序列时，每个位置都能关注到序列中的其他所有位置
####Multi-Head Attention（多头注意）

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726648779252.png)

__多头注意力__：通过并行地运行多个自注意力层，模型能够同时从不同的子空间表示中学习信息

###位置前馈网络
__正弦位置编码__：由于Transformer模型没有循环或卷积结构来捕获序列的顺序信息，因此引入了正弦和余弦函数的位置编码，以__提供关于序列中单词位置的信息__

###Positional Encoding
编入时序信息

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726649780009.png)

这个就是transformer输入时带入时序信息的方法

###实验

###评论

我们用来训练和评估模型的代码可以在https://github.com/ tensorflow/tensor2tensor上找到。