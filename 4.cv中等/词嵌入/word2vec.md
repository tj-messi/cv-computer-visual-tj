#word2vec

GloVe训练好的向量可视化：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107185753.png)

能从图标看出man和woman之间关系很大

##词预测

假设一个长度为m的句子，包含这些词：(w_1,w_2,w_3,..,w_m)，那么这个句子的概率（也就是这m个词共同出现的概率）是：P\left ( sen = (w_1,w_2,\cdots ,w_m) \right ) = P(w_1)P(w_2|w_1)P(w_3|w_2,w_1)\cdots P(w_m|w_{m-1}\cdots w_1)

一般来说，语言模型都是为了使得条件概率P(w_t|w_1,w_2,..,w_{t-1})最大化，不过考虑到近因效应，当前词只与距离它比较近的n个词更加相关(一般n不超过5，所以局限性很大)

##NNLM

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107185950.png)

就是做对下一个词的预测

##语言模型的训练

###构建映射矩阵(词嵌入矩阵)

先是获取大量文本数据(例如所有维基百科内容)

然后我们建立一个可以沿文本滑动的窗(例如一个窗里包含三个单词)

利用这样的滑动窗就能为训练模型生成大量样本数据

当这个窗口沿着文本滑动时，我们就能(真实地)生成一套用于模型训练的数据集。

不用多久，我们就能得到一个较大的数据集，从数据集中我们能看到在不同的单词组后面会出现的单词：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107191310.png)

###Word2Vec的两种架构：从CBOW到Skipgram模型

更进一步，为了更好的预测，其实不仅要考虑目标单词的前两个单词，还要考虑其后两个单词

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107191444.png)

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107191449.png)

『以上下文词汇预测当前词』架构被称为**连续词袋(CBOW)**

CBOW包括以下三层:

输入层：包含中个词的词向量，其中，表示单词的向量化表示函数，相当于此函数把一个个单词转化成了对应的向量化表示(类似one-hot编码似的)，表示上下文取的总词数，表示向量的维度；

投影层：将输入层的个向量做累加求和；

输出层：按理我们要通过确定的上下文决定一个我们想要的中心词，但怎么决定想要的中心词具体是  中的哪个呢？

通过计算各个可能中心词的概率大小，取概率最大的词便是我们想要的中心词，相当于是针对一个N维数组进行多分类，但计算复杂度太大，所以输出层改造成了一棵Huffman树，以语料中出现过的词当叶子结点，然后各个词出现的频率大小做权重

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107192046.png)

###还有另一种架构，刚好反过来，根据当前词推测当前单词可能的前后单词，这种架构就是所谓的Skipgram架构

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107192812.png)

##训练过程

具体做法是先创建两个矩阵：词嵌入Embedding矩阵(注意：这个 Embedding矩阵其实就是网络Onehot层到Embedding层映射的网络参数矩阵，所以使用Word Embedding等价于把Onehot层到Embedding层的网络用预训练好的参数矩阵初始化了)、上下文Context矩阵，这两个矩阵在我们的词汇表中嵌入了每个单词，且两个矩阵都有这两个维度

第一个维度，词典大小即vocab_size，比如可能10000，代表一万个词

第二个维度，每个词其嵌入的长度即embedding_size，比如300是一个常见值（当然，我们在前文也看过50的例子，比如上文1.1节中最后关于单词“king”的词嵌入长度）

**embedding size**的长度就是用来表示词的向量的长度

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107193136.png)

训练的过程还是这个标准套路/方法，比如

第一步，先用随机值初始化这些矩阵。在每个训练步骤中，我们采取一个相邻的例子及其相关的非相邻例子


具体而言，针对这个例子：“Thou shalt not make a machine in the likeness of a human mind”，我们来看看我们的第一组（对于not 的前后各两个邻居单词分别是：Thou shalt 、make a）：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107194144.png)

现在有四个单词：输入单词not，和上下文单词：thou（实际邻居词），aaron和taco（负面例子）
我们继续查找它们的嵌入
对于输入词not，我们查看Embedding矩阵
对于上下文单词，我们查看Context矩阵

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107194338.png)

第二步，计算输入嵌入与每个上下文嵌入的点积
还记得点积的定义否
两个向量a = [a1, a2,…, an]和b = [b1, b2,…, bn]的点积定义为：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107194600.png)

而这个点积的结果意味着『输入』和『上下文各个嵌入』的各自相似性程度，结果越大代表越相似

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107194911.png)

为了将这些分数转化为看起来像概率的东西——比如正值且处于0到1之间，可以通过sigmoid这一逻辑函数转换下

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107194954.png)

可以看到taco得分最高，aaron最低，无论是sigmoid操作之前还是之后。

第三步，既然未经训练的模型已做出预测，而且我们拥有真实目标标签来作对比，接下来便可以计算模型预测中的误差了，即让目标标签值减去sigmoid分数，得到所谓的损失函数

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107195030.png)

error = target - sigmoid_scores

第四步，我们可以利用这个错误分数来调整not、thou、aaron和taco的嵌入，使下一次做出这一计算时，结果会更接近目标分数

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107195215.png)

训练步骤到此结束，我们从中得到了这一步所使用词语更好一些的嵌入（not，thou，aaron和taco）

第五步，针对下一个相邻样本及其相关的非相邻样本再次执行相同的过程

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107195240.png)

当我们循环遍历整个数据集多次时，嵌入会继续得到改进。然后我们就可以停止训练过程，丢弃Context矩阵，并使用Embeddings矩阵作为下一项任务的已被训练好的嵌入