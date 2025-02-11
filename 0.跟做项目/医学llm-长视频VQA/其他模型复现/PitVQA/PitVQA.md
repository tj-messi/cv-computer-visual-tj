#PitVQA

##代码git clone

直接下载然后上传即可

##数据集

在PitVQA_dataset目录里面

下载github的数据集

解压到如下目录

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1739281388289.png)

原数据集是video-25

先video-to-image

然后preprocess image处理好了内容

##流程跑通

记得几个load的pretrain模型，都下载下来然后换成本地路径

然后train

	python main.py --dataset=pit24 --epochs=60 --batch_size=64 --lr=0.00002

##创新点复现