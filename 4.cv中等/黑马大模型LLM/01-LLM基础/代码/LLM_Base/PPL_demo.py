import math
import math
# 定义语料库
sentences = [
['I', 'have', 'a', 'pen'],
['He', 'has', 'a', 'book'],
['She', 'has', 'a', 'cat']
]
# 定义语言模型
unigram = {'I': 1/12, 'have': 1/12, 'a': 3/12, 'pen': 1/12,
           'He': 1/12,'has': 2/12,'book': 1/12,'She': 1/12, 'cat': 1/12}


perplexity = 0
for sentence in sentences:
    sentence_prob = 1
    for word in sentence:
        sentence_prob *= unigram[word]
    temp = -math.log(sentence_prob, 2)/len(sentence)
    perplexity+=2**temp
perplexity = perplexity/len(sentences)
print('困惑度为：', perplexity)
# 困惑度为： 8.15




# 困惑度为： 0.000325