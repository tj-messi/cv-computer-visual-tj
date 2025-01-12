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

	args = cfg.parse_args()

调用cfg.py中的parse_args拿到参数列args

	GPUdevice = torch.device('cuda', args.gpu_device)

拿GPU信息


	net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

	from hydra import initialize_config_module

	initialize_config_module("sam2_train", version_base="1.2")

调用utils.py的get network函数，确定是否进行分布式多GPU训练。
Hydra 是一个强大的配置管理框架，initialize_config_module 是其用来初始化配置模块的函数

之后会进入build_sam.py

	optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

这行代码创建了一个 Adam 优化器，用于优化模型 net 的参数。具体配置如下：

学习率：使用 args.lr（外部传入的值）。
动量参数：beta1=0.9 和 beta2=0.999，分别控制一阶和二阶动量。
防止分母为 0：设置了 eps=1e-08。
不使用权重衰减：weight_decay=0。
不启用 AMSGrad：amsgrad=False。
最终，优化器将在训练过程中逐步调整模型的参数，最小化损失函数，从而提高模型的性能。

	
	 '''load pretrained model'''

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)