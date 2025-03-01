#nnUNet

##数据准备

png格式的内容可以考虑参考其中的road模式的放置来安排数据集

##格式准备

我们退回到文件夹中，在nnUNet文件夹中创建一个新文件夹nnUNetFrame（该文件夹名称可以自拟）

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20250302000032.png)

在新文件夹中创建下列三个文件夹，文件夹名称固定（raw存放原始数据集，preprocessed存放预处理后的训练计划，results存放训练结果等，个人理解，如有错误可在评论区指正）

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20250302000154.png)

(建议大家创建一个test.py文件，存放常用的指令，比如环境变量和后续的训练命令等)

每次重新打开该项目进行训练或其他处理时，都需要先设置环境变量（复制粘贴到pycharm终端运行，建议对环境、库、包管理在anaconda prompt中进行，对项目管理都在pycharm中进行）

##设计环境变量
	
	# 设置环境变量，指向你的数据文件夹
	export nnUNet_raw='/media/tongji/nnUNet-master/zjz-nnUNetFrame/nnUNet_raw'
	export nnUNet_preprocessed='/media/tongji/nnUNet-master/zjz-nnUNetFrame/nnUNet_preprocessed'
	export nnUNet_results='/media/tongji/nnUNet-master/zjz-nnUNetFrame/nnUNet_results'

##数据格式转换




