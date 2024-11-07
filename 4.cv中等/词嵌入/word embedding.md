#word embedding

##词嵌入

机器理解文本的方式就是转化为向量

在NLP(自然语言处理)领域，文本表示是第一步，也是很重要的一步，通俗来说就是把人类的语言符号转化为机器能够进行计算的数字

文本表示分为**离散表示**和**分布式表示**：

##离散表示

###One-hot 向量

One-hot简称读热向量编码，也是特征工程中最常用的方法。其步骤如下：

构造文本分词后的字典，每个分词是一个比特值，比特值为0或者1。
每个分词的文本表示为该分词的比特位为1，其余位为0的矩阵表示。

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730380203070.png)

问题就是语料库增加这样会产生一个很稀疏维度很高的矩阵

这种表示方法的分词顺序和在句子中的顺序是无关的，不能保留词与词之间的关系信息

###词袋模型

词袋模型(Bag-of-words model)，像是句子或是文件这样的文字可以用一个袋子装着这些词的方式表现，这种表现方式不考虑文法以及词的顺序。

文档的向量表示可以直接将各词的词向量表示加和。例如：

John likes to watch movies. Mary likes too

John also likes to watch football games.

以上两句可以构造一个词典，**{"John": 1, "likes": 2, "to": 3, "watch": 4, "movies": 5, "also": 6, "football": 7, "games": 8, "Mary": 9, "too": 10} **

那么第一句的向量表示为：[1,2,1,1,1,0,0,0,1,1]，其中的2表示likes在该句中出现了2次，依次类推。

词袋模型同样有一下缺点：

词向量化后，词与词之间是有大小关系的，不一定词出现的越多，权重越大。
词与词之间是没有顺序关系的。

###TF-IDF

**TF-IDF**（term frequency–inverse document frequency）是一种用于信息检索与数据挖掘的常用加权技术。TF意思是词频(Term Frequency)，IDF意思是逆文本频率指数(Inverse Document Frequency)。

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730466416076.png)

分母之所以加1，是为了避免分母为0。

那么，，从这个公式可以看出，当w在文档中出现的次数增大时，而TF-IDF的值是减小的，所以也就体现了以上所说的了。

**缺点：**还是没有把词与词之间的关系顺序表达出来。

###n-gram模型

n-gram模型为了保持词的顺序，做了一个滑窗的操作，这里的n表示的就是滑窗的大小，例如2-gram模型，也就是把2个词当做一组来处理，然后向后移动一个词的长度，再次组成另一组词，把这些生成一个字典，按照词袋模型的方式进行编码得到结果。改模型考虑了词的顺序。

例如：

John likes to watch movies. Mary likes too

John also likes to watch football games.

以上两句可以构造一个词典，{"John likes”: 1, "likes to”: 2, "to watch”: 3, "watch movies”: 4, "Mary likes”: 5, "likes too”: 6, "John also”: 7, "also likes”: 8, “watch football”: 9, "football games": 10}

那么第一句的向量表示为：[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]，其中第一个1表示John likes在该句中出现了1次，依次类推。

**缺点：**随着n的大小增加，词表会成指数型膨胀，会越来越大。

###2.5离散表示的问题

由于存在以下的问题，对于一般的NLP问题，是可以使用离散表示文本信息来解决问题的，但对于要求精度较高的场景就不适合了。

无法衡量词向量之间的关系。

词表的维度随着语料库的增长而膨胀。

n-gram词序列随语料库增长呈指数型膨胀，更加快。

离散数据来表示文本会带来数据稀疏问题，导致丢失了信息，与我们生活中理解的信息是不一样的
