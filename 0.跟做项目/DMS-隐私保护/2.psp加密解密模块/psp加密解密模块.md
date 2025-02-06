#psp模块
##前期环境数据
git下来psp

	git clone https://github.com/eladrich/pixel2style2pixel.git

有些模块会报错无法引入，记得pip install 一下

其中 

	pip install dlib

会爆出很多错

先要安装好cmake环境

	sudo apt install cmake

然后安装好 C++ 编译器

	sudo apt install build-essential

都ok了之后要先找一下驾驶员数据

	https://driveandact.com/

这个网站里面大多是是黑白红外线摄像数据

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1737292756031.png)

	https://www.kaggle.com/c/state-farm-distracted-driver-detection/data

这个网站都是侧身的数据

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/a50ec17922fd76219c40a901241d89f.png)

	 https://www.dropbox.com/sh/yndzlk3o90ooq2j/AACWUT8xjabmILM6-rm1_gNAa?dl=0


##项目第一步：人脸特征提取&存储

	python scripts/store_latent_codes.py \
	--exp_dir=./exp-zjz/super_resolution_4 \
	--checkpoint_path=./pretrained_models/psp_celebs_super_resolution.pt \
	--data_path=./exp-zjz/driver-face \
	--test_batch_size=4 \
	--test_workers=4 >/dev/null &


当不输入那个 
	
	>/dev/null &

此时就可以监视窗口