import os
import requests
import json

from openai import OpenAI

'''
2025-2-12
图片路径:/data/coding/PitVQA-main/PitVQA_dataset/images/video_xx/xxxxx.png
QA对路径:/data/coding/PitVQA-main/PitVQA_dataset/qa-classification/video_xx/xxxxx.txt
可以把整个txt文件夹的QA对加载成描述性文本作为输入的提示
拿到返回的CoT之后可以修改之前数据集里面的Q|A|L
'''

# 设置你的 DeepSeek API 密钥
api_key = "sk-1f270af835a0429ca31b8d93ceee3387"
base_url = "https://api.deepseek.com"


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

# 上传图像并获取推理结果
def generate_cot_from_image_and_qa(image_file_path, qa_pairs,prompt_text):
    # 上传图像文件
    with open(image_file_path, 'rb') as image_file:
        files = {
            'image': image_file,
        }
        headers = {
            'Authorization': f'Bearer {api_key}'
        }

        # 上传图像
        response = requests.post(f"{base_url}/upload", headers=headers, files=files)
        
        if response.status_code == 200:
            image_url = response.json()['image_url']
            print(f"Image uploaded successfully: {image_url}")
        else:
            print(f"Failed to upload image. Status Code: {response.status_code}")
            return None

    # 生成推理链（CoT）
    payload = {
        'image_url': image_url,
        'image_description': prompt_text,
        'qa_pairs': qa_pairs,
    }
    response = requests.post(f"{base_url}/generate_cot", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()['cot']  # 返回生成的推理链
    else:
        print(f"Failed to generate CoT. Status Code: {response.status_code}")
        return None



def main():

    # 图片路径
    image_path = '/data/coding/PitVQA-main/PitVQA_dataset/images/video_01/00019.png'

    # 文本路径
    txt_path = '/data/coding/PitVQA-main/PitVQA_dataset/qa-classification/video_01/00019.txt'

    prompt_text , QA_pairs = merge_qa_pairs(txt_path)
    # print(prompt_text)
    # print(QA_pairs)

    # 调用函数并获取推理链
    cot = generate_cot_from_image_and_qa(image_path, QA_pairs,prompt_text)

    if cot:
        print("Generated Chain of Thought:")
        print(cot)

if __name__ == '__main__':
    main()