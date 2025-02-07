#Surgical-GPT

##跑通流程

###git

无问题

###data下载

对应github上的data下载处理

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1738932443290.png)

###整理源数据集合的处理方式

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1738934115549.png)

数据排列的方式

###train

	python train.py --lr=0.00001 --checkpoint_dir='checkpoints/efvlegpt2Swin/m18_v1_z_qf_' --dataset_type='m18' --tokenizer_ver='btv2' --model_ver='efvlegpt2Swin' --model_subver='v1' --vis_pos_emb='zeroes'


##创新点复现