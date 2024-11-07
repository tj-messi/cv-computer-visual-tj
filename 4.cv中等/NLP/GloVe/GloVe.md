#GloVe

##简述

正如GloVe论文的标题而言，**GloVe的全称叫Global Vectors for Word Representation，它是一个基于全局词频统计（count-based & overall statistics）的词表征（word representation）工具，它可以把一个单词表达成一个由实数组成的向量，这些向量捕捉到了单词之间一些语义特性，比如相似性（similarity）、类比性（analogy）等。**我们通过对向量的运算，比如欧几里得距离或者cosine相似度，可以计算出两个单词之间的语义相似性。

##GloVe实现

###构建共现矩阵

局域窗中的word-word共现矩阵可以挖掘语法和语义信息，例如：

I like deep learning.

I like NLP.

I enjoy flying

有以上三句话，设置滑窗为2，可以得到一个词典：{"I like","like deep","deep learning","like NLP","I enjoy","enjoy flying","I like"}。

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107212633.png)

根据语料库（corpus）构建一个共现矩阵（Co-ocurrence Matrix）X，矩阵中的每一个元素 Xij 代表单词 i 和上下文单词 j 在特定大小的上下文窗口（context window）内共同出现的次数。一般而言，这个次数的最小单位是1，但是GloVe不这么认为：它根据两个单词在上下文窗口的距离 d，提出了一个衰减函数（decreasing weighting）：decay=1/d 用于计算权重，也就是说距离越远的两个单词所占总计数（total count）的权重越小

###词向量和共现矩阵的近似关系
构建词向量（Word Vector）和共现矩阵（Co-ocurrence Matrix）之间的近似关系，论文的作者提出以下的公式可以近似地表达两者之间的关系：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107212855.png)

其中，是我们最终要求解的词向量；分别是两个词向量的bias term。当然你对这个公式一定有非常多的疑问，比如它到底是怎么来的，为什么要使用这个公式，为什么要构造两个词向量 ？请参考文末的参考文献。

损失函数：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730986543050.png)

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730986632647.png)

Xi就是共现矩阵的行总和

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730986693153.png)

Pi，k 表示单词k出现在单词i语境中的概率

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730986746134.png)

两个条件概率的比率

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730986842171.png)

此时词向量vi，vj，vk通过函数计算ratio，如果也能得到一样的规律就可以发现词向量和共现矩阵之间的关系，先假设这个函数是g(vi,vj,vk)

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730987246662.png)

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730987281541.png)

这个g函数的探索如下

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730987369908.png)