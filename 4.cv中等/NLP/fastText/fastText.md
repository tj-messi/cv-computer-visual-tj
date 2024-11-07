#fastText

论文：

Probabilistic FastText for Multi-Sense Word Embeddings


##简单介绍

**需要拿到**

英语单词通常有其内部结构和形成⽅式。例如，我们可以从“dog”“dogs”和“dogcatcher”的字⾯上推测它们的关系。这些词都有同⼀个词根“dog”，但使⽤不同的后缀来改变词的含义。而且，这个关联可以推⼴⾄其他词汇。

在word2vec中，我们并没有直接利⽤构词学中的信息。⽆论是在跳字模型还是连续词袋模型中，我们都将形态不同的单词⽤不同的向量来表⽰。例如，“dog”和“dogs”分别⽤两个不同的向量表⽰，而模型中并未直接表达这两个向量之间的关系。鉴于此，fastText提出了⼦词嵌⼊**(subword embedding)**的⽅法，从而试图将**构词信息**引⼊word2vec中的CBOW

这里有一点需要特别注意，一般情况下，使用fastText进行文本分类的同时也会产生词的embedding，即embedding是fastText分类的产物。除非你决定使用预训练的embedding来训练fastText分类模型，这另当别论

##n-gram表示单词

word2vec把语料库中的每个单词当成原子的，它会为每个单词生成一个向量。这忽略了单词内部的形态特征，比如：“book” 和“books”，“阿里巴巴”和“阿里”，这两个例子中，两个单词都有较多公共字符，即它们的内部形态类似，但是在传统的word2vec中，这种单词内部形态信息因为它们被转换成不同的id丢失了

为了克服这个问题，fastText使用了字符级别的n-grams来表示一个单词。

	对于单词“book”，假设n的取值为3，则它的trigram有:

	“<bo”, “boo”, “ook”, “ok>”

其中，<表示前缀，>表示后缀。于是，我们可以用这些trigram来表示“book”这个单词，进一步，我们可以用这4个trigram的向量叠加来表示“apple”的词向量。

这带来两点好处：

1.对于低频词生成的词向量效果会更好。因为它们的n-gram可以和其它词共享。

2.对于训练词库之外的单词，仍然可以构建它们的词向量。我们可以叠加它们的字符级n-gram向量。

##fasttext架构

之前提到过，fastText模型架构和word2vec的CBOW模型架构非常相似。下面是fastText模型架构图

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20241107201617.png)

注意：此架构图没有展示词向量的训练过程。可以看到，和CBOW一样，fastText模型也只有三层：输入层、隐含层、输出层（Hierarchical Softmax），输入都是多个经向量表示的单词，输出都是一个特定的target，隐含层都是对多个词向量的叠加平均。

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730982693356.png)

不同的是，

CBOW的输入是目标单词的上下文，fastText的输入是多个单词及其n-gram特征，这些特征用来表示单个文档；

CBOW的输入单词被one-hot编码过，fastText的输入特征是被embedding过；

CBOW的输出是目标词汇，fastText的输出是文档对应的类标。
**值得注意的是，fastText在输入时，将单词的字符级别的n-gram向量作为额外的特征；在输出时，fastText采用了分层Softmax，大大降低了模型训练时间。**这两个知识点在前文中已经讲过，这里不再赘述

##fasttext核心思想

fastText的核心思想就是：将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类

这中间涉及到两个技巧：字符级n-gram特征的引入以及分层Softmax分类

##效果

假设我们有两段文本：

肚子 饿了 我 要 吃饭

肚子 饿了 我 要 吃东西

这两段文本意思几乎一模一样，如果要分类，肯定要分到同一个类中去。但在传统的分类器中，用来表征这两段文本的向量可能差距非常大。传统的文本分类中，你需要计算出每个词的权重，比如TF-IDF值， “吃饭”和“吃东西” 算出的TF-IDF值相差可能会比较大，其它词类似，于是，VSM（向量空间模型）中用来表征这两段文本的文本向量差别可能比较大

但是fastText就不一样了，它是用单词的embedding叠加获得的文档向量，词向量的重要特点就是向量的距离可以用来衡量单词间的语义相似程度，于是，在fastText模型中，这两段文本的向量应该是非常相似的，于是，它们很大概率会被分到同一个类中。

使用词embedding而非词本身作为特征，这是fastText效果好的一个原因；另一个原因就是字符级n-gram特征的引入对分类效果会有一些提升

## fastText与Word2Vec的不同

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730985017858.png)

word2vec的目的是得到词向量，该词向量最终是在输入层得到的，输出层对应的h-softmax也会生成一系列的向量，但是最终都被抛弃，不会使用。

fastText则充分利用了h-softmax的分类功能，遍历分类树的所有叶节点，找到概率最大的label

## 实现

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730985351628.png)
	
	from gensim.models import FastText
	from gensim.models.word2vec import LineSentence
	
	model=FastText(LineSentence(open(r'../GloVe/data/test.txt', 'r', encoding='utf8')),
	               vector_size=150, window=5, min_count=5, epochs=10, min_n=3, max_n=6, word_ngrams=1,workers=4)

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1730985384572.png)