import os
import requests
import json

import openai

'''
2025-2-12
图片路径:/data/coding/PitVQA-main/PitVQA_dataset/images/video_xx/xxxxx.png
QA对路径:/data/coding/PitVQA-main/PitVQA_dataset/qa-classification/video_xx/xxxxx.txt
可以把整个txt文件夹的QA对加载成描述性文本作为输入的提示
拿到返回的CoT之后可以修改之前数据集里面的Q|A|L
'''


# 合并问答对
def merge_qa_pairs(txt_path):
    merged_text = "Based on the image, here is a description of the surgical procedure and related details:\n"
    QA_pairs = []
    with open(txt_path,'r') as file:
        for line in file : 
            line = line.strip()
            question = line.split('|')[0]
            answer = line.split('|')[1]
            QA_pairs.append((question, answer))
            merged_text += f"Q: {question} A: {answer} \n"
    
    
    return merged_text,QA_pairs



def main():
    # 设置你的 DeepSeek API 密钥
    api_key = "sk-d8GxjUNttjCpE65ACUovC8DTWHB968fdMktPveAOHVxHSLRf"
    base_url = "https://api.deepseek.ai"

    # 图片路径
    image_path = '/data/coding/PitVQA-main/PitVQA_dataset/images/video_01/00019.png'

    # 文本路径
    txt_path = '/data/coding/PitVQA-main/PitVQA_dataset/qa-classification/video_01/00019.txt'

    prompt_text , QA_pairs = merge_qa_pairs(txt_path)
    # print(prompt_text)
    # print(QA_pairs)

    # 打开图像文件并上传
    with open(image_path, "rb") as image_file:
        # 构建CoT提示，将QA对与图像描述结合
        prompt = f"""
        I have an image and multiple QA pairs. For each QA pair below, provide a Chain of Thought (CoT) to explain how the answer is derived based on the image and the context of the QA pair.
        
        Image Description: {prompt_text}

        QA Pairs:
        """

        # 添加每一个问答对到提示中
        for question, answer in QA_pairs:
            prompt += f"Q: {question}\nA: {answer}\n\n"

        prompt += "Step-by-step Chain of Thought for each QA pair:\n"

        # 调用 OpenAI API 生成 CoT
        response = openai.Completion.create(
            model="text-davinci-003",  # 或使用其他适当的模型
            prompt=prompt,
            max_tokens=1000,  # 根据需要调整 token 长度
            temperature=0.7  # 温度控制推理的随机性
        )

    # 打印生成的CoT
    print(response.choices[0].text.strip())

if __name__ == '__main__':
    main()