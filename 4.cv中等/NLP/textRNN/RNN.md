#RNN

循环神经网络（Recurrent Neural Network, RNN）是一类以序列（sequence）数据为输入，在序列的演进方向进行递归（recursion）且所有节点（循环单元）按链式连接的递归神经网络（recursive neural network

##RNN网络结构

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107223740.png)

现在我们考虑输⼊数据存在时间相关性的情况。假设

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107224250.png)

 是序列中时间步t的小批量输⼊，

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107224304.png)

 是该时间步的隐藏变量。那么根据以上结构图当前的隐藏变量的公式如下

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107224315.png)

从以上公式我们可以看出，这⾥我们保存上⼀时间步的隐藏变量

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107224824.png)

并引⼊⼀个新的权重参数，该参数⽤来描述在当前时间步如何使⽤上⼀时间步的隐藏变量。具体来说，时间步 t 的隐藏变量的计算由当前时间步的输⼊和上⼀时间步的隐藏变量共同决定。

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107224903.png) 函数其实就是激活函数。

我们在这添加了

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107225624.png)

由于隐藏状态在当前时间步的定义使⽤了上⼀时间步的隐藏状态，上式的计算是循环的。使⽤循环计算的⽹络即循环神经⽹络（recurrent neural network）

在时间步t，输出层的输出和多层感知机中的计算类似：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107225659.png)

##双向RNN
之前介绍的循环神经⽹络模型都是假设当前时间步是由前⾯的较早时间步的序列决定的，因此它 们都将信息通过隐藏状态从前往后传递。有时候，当前时间步也可能由后⾯时间步决定。例如， 当我们写下⼀个句⼦时，可能会根据句⼦后⾯的词来修改句⼦前⾯的⽤词。**双向循环神经⽹络通过增加从后往前传递信息的隐藏层来更灵活地处理这类信息。**下图演⽰了⼀个含单隐藏层的双向循环神经⽹络的架构

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107225934.png)

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730991561563.png)

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730992173053.png)

##BPTT （时间反向传播算法）

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107231529.png)

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1731031000772.png)