#psp模块

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

	https://www.kaggle.com/c/state-farm-distracted-driver-detection/data

	