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

