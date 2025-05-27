#Chain-of-Though

##参考论文

Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

##改进GPT2的方法

采用思考链，修改改进dataloader里面的如下部分

       # question and answer
       # How many instruments are present in the image?|one
        
       question = self.vqas[idx][1].split('|')[0]
       answer = self.vqas[idx][1].split('|')[1]
       label = self.labels.index(str(answer))

这里把一段这样的数据集

How many instruments are present in the image?|one

分割成为如下两段

Q：How many instruments are present in the image?

A：one

那么我们可以增加一段思考链的Answer，来改进QA对

新的数据集就是这样的格式

How many instruments are present in the image?|i can only see one freer_elevator in the centre of the image so the answer is one|one


接下来针对我们的数据去做

	https://blog.csdn.net/godnightshao/article/details/130455465?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522606827386e52a55bf4e3618e46574854%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=606827386e52a55bf4e3618e46574854&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-130455465-null-null.142^v101^pc_search_result_base5&utm_term=%E8%B0%83%E7%94%A8openAI%E7%9A%84api&spm=1018.2226.3001.4187


程序如下

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
	
	# 初始化客户端，需替换为实际部署的API地址和密钥
	client = OpenAI(
	    api_key = "sk-d8GxjUNttjCpE65ACUovC8DTWHB968fdMktPveAOHVxHSLRf",  
	    base_url="https://tbnx.plus7.plus/v1"  # 替换为你的API地址
	)
	
	
	# 合并问答对
	def merge_qa_pairs(txt_path):
	    merged_text = "here are some description of the surgical procedure and related details:\n"
	    QA_pairs = ""
	    with open(txt_path,'r') as file:
	        for line in file : 
	            line = line.strip()
	            question = line.split('|')[0]
	            answer = line.split('|')[1]
	            QA_pairs += f"Q: {question} A: {answer} \n"
	            merged_text += f"Q: {question} A: {answer} \n"
	    
	    merged_text += 'Here are some Q-A pairs , i want you to get the chain-of-though of every Q-A pairs based on the Q-A pairs and the description\n'
	    merged_text += 'list the CoT like this format:\n'
	    merged_text += 'CoT1:......\n'
	    merged_text += 'CoT2:......\n'
	    merged_text += 'CoT3:......\n'
	
	    return merged_text,QA_pairs
	
	
	
	
	def main():
	
	    # 图片路径
	    image_path = '/data/coding/PitVQA-main/PitVQA_dataset/images/video_01/00019.png'
	
	    # 文本路径
	    txt_path = '/data/coding/PitVQA-main/PitVQA_dataset/qa-classification/video_01/00019.txt'
	
	    prompt_text , QA_pairs = merge_qa_pairs(txt_path)
	    # print(prompt_text)
	    # print(QA_pairs)
	
	    # 构造对话请求
	    response = client.chat.completions.create(
	        model="deepseek-reasoner",  # R1模型标识符[[1,2,15]]
	        messages=[
	            {"role": "system", "content": "你是一个专业的手术医生"},
	            {"role": "user", "content": prompt_text},
	            {"role": "user", "content": f"QA Pairs: {QA_pairs}"}
	        ],
	        temperature=0.7,
	        stream=False  # 如需流式响应可设为True
	    )
	
	    # 输出响应结果
	    print(response.choices[0].message.content)
	
	if __name__ == '__main__':
	    main()