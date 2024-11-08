#LSTM

##LSTM简介

在你阅读这篇文章时候，你都是基于自己已经拥有的对先前所见词的理解来推断当前词的真实含义。我们不会将所有的东西都全部丢弃，然后用空白的大脑进行思考。我们的思想拥有持久性。LSTM就是具备了这一特性。

这篇将介绍另⼀种常⽤的⻔控循环神经⽹络：**⻓短期记忆（long short-term memory，LSTM）[1]。**它⽐⻔控循环单元的结构稍微复杂⼀点，也是为了解决在RNN网络中梯度衰减的问题，是GRU的一种扩展。

可以先理解GRU的过程，在来理解LSTM会容易许多，链接地址：三步理解--门控循环单元(GRU)

LSTM 中引⼊了3个⻔，即输⼊⻔（input gate）、遗忘⻔（forget gate）和输出⻔（output gate），以及与隐藏状态形状相同的记忆细胞（某些⽂献把记忆细胞当成⼀种特殊的隐藏状态），从而记录额外的信息。

##三门

与⻔控循环单元中的重置⻔和更新⻔⼀样，⻓短期记忆的⻔的输⼊均为当前时间步输⼊Xt与上⼀时间步隐藏状态Ht−1，输出由激活函数为sigmoid函数的全连接层计算得到。如此⼀来，这3个⻔元素的值域均为[0, 1]。如下图所示：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1731036959124.png)

##候选记忆细胞

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1731037197953.png)

##记忆细胞

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1731043401631.png)

##隐藏状态

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1731044143955.png)