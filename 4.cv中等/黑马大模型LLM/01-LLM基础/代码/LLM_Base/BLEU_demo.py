# 第一步安装nltk的包-->pip install nltk
from nltk.translate.bleu_score import sentence_bleu

def cumulative_bleu(reference, candidate):

    bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu_2_gram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu_3_gram = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu_4_gram = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram
    # return bleu_1_gram, bleu_2_gram

# 生成文本
candidate_text = ["This", "is",  "some",  "generated", "text"]

# 参考文本列表
reference_texts = [["This", "is",  "a",  "reference", "text"],
                   ["This", "is",  "another", "reference", "text"]]

# 计算 Bleu 指标
c_bleu = cumulative_bleu(reference_texts, candidate_text)

# 打印结果

print("The Bleu score is:", c_bleu)
