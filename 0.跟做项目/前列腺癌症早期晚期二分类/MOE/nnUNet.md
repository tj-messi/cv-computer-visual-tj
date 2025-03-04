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

可以按照road的方式参考，不过注意前列腺超声影像为RGBA格式的数据需要转换

	def load_and_convert_case(input_image: str, input_seg: str, output_image: str, output_seg: str,
	                          min_component_size: int = 50):
	    seg = io.imread(input_seg)
	    seg[seg == 255] = 1
	    image = io.imread(input_image)
	    # print(image.shape)
	    if image.shape[-1] == 4:
	        image = image[..., :3] 
	
	    image = image.sum(2)
	    mask = image == (3 * 255)
	    # the dataset has large white areas in which road segmentations can exist but no image information is available.
	    # Remove the road label in these areas
	    mask = generic_filter_components(mask, filter_fn=lambda ids, sizes: [i for j, i in enumerate(ids) if
	                                                                         sizes[j] > min_component_size])
	    mask = binary_fill_holes(mask)
	    seg[mask] = 0
	    seg = seg[..., 0]
	    print(seg.shape)
	    io.imsave(output_seg, seg, check_contrast=False)
	    image_pil = Image.fromarray(image.astype(np.uint8))  # 转换为 PIL 图像对象
	    image_pil = image_pil.convert('RGB')  # 确保是 RGB 图像
	    image_pil.save(output_image)  # 保存图像

确定mask的图像shape为（H，W）

确定了图像的格式为（H，W，4）的时候就可以只取前三个管道了

	image = image[...,:3]

##开始训练

先处理数据

	python Dataset1234_Prostate.py 

预处理，计划

	nnUNetv2_plan_and_preprocess -d 1234 --verify_dataset_integrity

训练，其中FOLD是0,1,2,3,4

	nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD

##接口test

	nnUNetv2_predict -i /media/tongji/Deeplabv3/train_output/test/predicted_mask_PB03900034.png  -o /media/tongji/nnUNet-master -d 1234 -c 2d --save_probabilities 

针对/media/tongji/Deeplabv3/train_output/test/predicted_mask_PB03900034.png

##注意

nnUNet对分割图片保存了二值化

所以要进行转化

	import numpy as np
	from skimage import io
	
	def convert_mask(input_mask_path: str, output_mask_path: str):
	    # 读取输入的分割图像
	    mask = io.imread(input_mask_path)
	    
	    # 确保是二值图像（0 或 1）
	    mask = (mask > 0).astype(np.uint8)  # 处理任何非零值为1，零值保持为0
	    
	    # 将0和1转换为0和255
	    mask = mask * 255
	    
	    # 保存转换后的分割图像
	    io.imsave(output_mask_path, mask)
	    print(f"转换后的掩膜保存为: {output_mask_path}")
	
	# 示例：使用转换函数
	input_mask = "/media/tongji/nnUNet-master/zjz-nnUNetFrame/nnUNet_results/Dataset1234_Prostate/nnUNetTrainer__nnUNetPlans__2d/fold_1/validation/img-4293.png"  # 输入的掩膜路径
	output_mask = "/media/tongji/nnUNet-master/zjz-nnUNetFrame/nnUNet_results/Dataset1234_Prostate/nnUNetTrainer__nnUNetPlans__2d/fold_1/validation/img-4293-0to255.png"  # 输出的掩膜路径
	
	convert_mask(input_mask, output_mask)