#Surgical-GPT

##跑通流程

###git

无问题

###data下载

对应github上的data下载处理

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1738932443290.png)

###整理源数据集合的处理方式

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1738934115549.png)
-
cholec-80：

	https://www.kaggle.com/datasets/mohaddesehkz/cholec80

endovis：

	https://endovissub2017-workflow.grand-challenge.org/Data/

psi-ava:

github主页

	https://github.com/BCV-Uniandes/TAPIR

google-driver

	https://drive.google.com/file/d/1NVzhRnqy4A9W1Hj17r-v8cO_VnfKeVf_/view?pli=1

	$ wget http://157.253.243.19/PSI-AVA/PSI-AVA.tar.gz
	$ tar -xzvf PSI-AVA.tar.gz

数据排列的方式都是视频or帧数据作一个文件夹
然后其中的VQA-annotation在另一个文件夹

###train

	python train.py --lr=0.00001 --checkpoint_dir='checkpoints/efvlegpt2Swin/m18_v1_z_qf_' --dataset_type='c80' --tokenizer_ver='btv2' --model_ver='efvlegpt2Swin' --model_subver='v1' --vis_pos_emb='zeroes'

其中model.load网上的模型不能成功需要下载下来到本地使用

参考csdn上的本地模型解决方法

把dataloaderGPT2Classification.py中108行

	self.image_processor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

改为现在模型的相对路径
	
	self.image_processor = AutoFeatureExtractor.from_pretrained("./model_pt_dir")

这个相对路径里面保存着模型

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1738939317937.png)

gpt-2也需要如法炮制

###虚拟环境配置

可以直接遇到什么问题pip 什么环境

##断点debug

先初始化tokenizer

然后开始找train和val集合

        elif args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
            
            train_dataset = Cholec80VQAGPTClassification(train_seq, folder_head, folder_tail, model_ver=args.model_ver)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=8)
            val_dataset = Cholec80VQAGPTClassification(val_seq, folder_head, folder_tail, model_ver=args.model_ver)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False, num_workers=8)

这里会直接进入/Surgical-GPT/dataloaders/Cholec80VQAGPTClassification

他会载入pretrain好的swin-Transformers数据集

里面会读取所有的filename和qa对

然后设置label

	# labels
	        self.labels = ['no', 'calot triangle dissection', 'yes', '1', '2', 'gallbladder dissection', 
	                        'clipping cutting', 'gallbladder retraction', '0', 'cleaning coagulation', 
	                        'gallbladder packaging', 'preparation', '3']

然后设置好train和val集之后核定class_num

	args.num_class = 13

然后debug到如下报错

	FileNotFoundError: [Errno 2] No such file or directory: 'dataset/Cholec80-VQA/cropped_image/7/6200.png'

这时候等待数据上传

##创新点复现

