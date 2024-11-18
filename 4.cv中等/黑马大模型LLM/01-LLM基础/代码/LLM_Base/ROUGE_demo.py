# 第一步：安装rouge-->pip install rouge
from rouge import Rouge

# 生成文本
generated_text = "This is some generated text."

# 参考文本列表
reference_texts = ["This is a reference text.", "This is another generated reference text."]

# 计算 ROUGE 指标
rouge = Rouge()
scores = rouge.get_scores(generated_text, reference_texts[1])
print(f'scores-->{scores}')

# 打印结果
print("ROUGE-1 precision:", scores[0]["rouge-1"]["p"])
print("ROUGE-1 recall:", scores[0]["rouge-1"]["r"])
print("ROUGE-1 F1 score:", scores[0]["rouge-1"]["f"])
# ROUGE-1 precision: 0.8
# ROUGE-1 recall: 0.6666666666666666
# ROUGE-1 F1 score: 0.7272727223140496