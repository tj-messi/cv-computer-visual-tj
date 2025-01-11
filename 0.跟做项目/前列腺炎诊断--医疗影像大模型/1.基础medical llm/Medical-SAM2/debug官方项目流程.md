#debug流程

##输入参数处理

输入带参数的调用命令

	python train_2d.py -net sam2 -exp_name REFUGE_MedSAM2 -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset REFUGE -data_path ./data/REFUGE

然后执行main开始：

	torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

这一行代码用于启用 自动混合精度（AMP, Automatic Mixed Precision） 计算，它结合了 PyTorch 的 autocast 功能，并明确指定了使用的设备类型（cuda，即 GPU）以及混合精度的数据类型（bfloat16）。



	if torch.cuda.get_device_properties(0).major >= 8:

判断GPU架构

获取第0号GPU（默认GPU）的设备属性。
返回一个 torch.cuda.DeviceProperties 对象，包含有关该GPU的详细信息。.major:返回GPU架构的主版本号。对于 NVIDIA Ampere架构，主版本号为 8（如 A100、RTX 30 系列等 GPU）。
对于较低版本的架构（如 Volta, Turing），主版本号小于 8。
>= 8:如果GPU架构版本为 Ampere 或更高，则执行后续代码，启用 TF32 模式。


	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True

启用TF32模式

