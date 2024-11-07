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