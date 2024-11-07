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