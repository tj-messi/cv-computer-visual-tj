#Video-MAE

##数据准备

fine-tunning数据准备

	frame_folder_path total_frames label
	
	# ssv2 rawframes data validation list
	your_path/SomethingV2/frames/74225 62 140
	your_path/SomethingV2/frames/116154 51 127
	your_path/SomethingV2/frames/198186 47 173
	your_path/SomethingV2/frames/137878 29 99
	your_path/SomethingV2/frames/151151 31 166

注意数据格式，我们的数据格式不一致，做一个

	create_csv.py


##fine_tine

使用蒸馏后的模型，base版本

	--model vit_base_patch16_224

走finetune路线

注意二分类并非多分类，把损失函数改成二分类的交叉熵即可