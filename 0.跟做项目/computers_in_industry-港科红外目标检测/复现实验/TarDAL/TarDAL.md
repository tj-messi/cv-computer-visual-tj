# TarDAL

## 数据准备

准备数据

TarDAL ROOT
├── data
|   ├── **m3fd**
|   |   ├── ir # infrared images
|   |   ├── vi # visible images
|   |   ├── labels # labels in txt format (yolo format)
|   |   └── meta # meta data, includes: pred.txt, train.txt, val.txt
|   ├── tno
|   |   ├── ir # infrared images
|   |   ├── vi # visible images
|   |   └── meta # meta data, includes: pred.txt, train.txt, val.txt
|   ├── roadscene
|   └── ...


## train前准备

需要使用WANDB来监管训练模型数据

网站
	
	https://wandb.ai/site


## 禁止调用pre_train模型

开始debug

###第一个u2net

在pipeline/saliency.py内

	def __init__(self, url: str):
	        # init device
	        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	        logging.info(f'deploy u2net on device {str(device)}')
	        self.device = device
	
	        # init u2net small (u2netp)
	        net = U2NETP(in_ch=1, out_ch=1)
	        logging.info(f'init u2net small model with (1 -> 1)')
	        self.net = net
	
	        # download pretrained parameters
	        ckpt_p = Path.cwd() / 'weights' / 'v1' / 'u2netp.pth'
	        logging.info(f'download pretrained u2net weights from {url}')
	        socket.setdefaulttimeout(5)
	        try:
	            logging.info(f'starting download of pretrained weights from {url}')
	            ckpt = torch.hub.load_state_dict_from_url(url, model_dir=ckpt_p.parent, map_location='cpu')
	        except Exception as err:
	            logging.fatal(f'load {url} failed: {err}, try download pretrained weights manually')
	            sys.exit(1)
	        
	        # u2net init cancel -> random zjz
	        # net.load_state_dict(ckpt)
	
	        logging.info(f'load pretrained u2net weights from {str(ckpt_p)}')
	
	        # move to device
	        net.to(device)
	
	        # more parameters
	        self.transform_fn = Compose([Resize(size=(320, 320)), Normalize(mean=0.485, std=0.229)])

里面load了一个u2net的模型

	net.load_state_dict(ckpt)

注释掉

###vgg16

在/pipeline/iqa.py中

	class IQA 的初始化load了一个预训练模型
	
	    def __init__(self, url: str):
	        # init device
	        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	        logging.info(f'deploy iqa on device {str(device)}')
	        self.device = device
	
	        # init vgg backbone
	        extractor = vgg16().features
	        logging.info(f'init iqa extractor with (3 -> 1)')
	        self.extractor = extractor
	
	        # download pretrained parameters
	        ckpt_p = Path.cwd() / 'weights' / 'v1' / 'iqa.pth'
	        logging.info(f'download pretrained iqa weights from {url}')
	        socket.setdefaulttimeout(5)
	        try:
	            logging.info(f'starting download of pretrained weights from {url}')
	            ckpt = torch.hub.load_state_dict_from_url(url, model_dir=ckpt_p.parent, map_location='cpu')
	        except Exception as err:
	            logging.fatal(f'load {url} failed: {err}, try download pretrained weights manually')
	            sys.exit(1)
	        # load -> random zjz
	        # extractor.load_state_dict(ckpt)
	        logging.info(f'load pretrained iqa weights from {str(ckpt_p)}')
	
	        # move to device
	        extractor.to(device)
	
	        # more parameters
	        self.transform_fn = Compose([Resize((672, 672)), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	        self.upsample = Resize((672, 672))

其中

	extractor.load_state_dict(ckpt)

注释掉

##TarDAL-model

在/pipeline/fuse.py

中

        if f_ckpt is not None:
            if 'http' in f_ckpt:
                ckpt_p = Path.cwd() / 'weights' / 'v1' / 'tardal.pth'
                url = f_ckpt
                logging.info(f'download pretrained parameters from {url}')
                try:
                    ckpt = torch.hub.load_state_dict_from_url(url, model_dir=ckpt_p.parent, map_location='cpu')
                except Exception as err:
                    logging.fatal(f'connect to {url} failed: {err}, try download pretrained weights manually')
                    sys.exit(1)
            else:
                ckpt = torch.load(f_ckpt, map_location='cpu')
            # self.load_ckpt(ckpt)

的self.load_ckpt(ckpt)

在/pipeline/detect.py中

	class Detect:
	    r"""
	    Init detect pipeline to detect objects from fused images.
	    """
	
	    def __init__(self, config, mode: Literal['train', 'inference'], nc: int, classes: List[str], labels: List[Tensor]):
	        # attach hyper parameters
	        self.config = config
	        self.mode = mode  # fuse computation mode: train(grad+graph), eval(graph), inference(x)
	
	        # init device
	        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	        logging.info(f'deploy {config.detect.model} on device {str(device)}')
	        self.device = device
	
	        # init yolo model
	        model_t = config.detect.model
	        config_p = Path(__file__).parents[1] / 'module' / 'detect' / 'models' / f'{model_t}.yaml'
	        net = Model(cfg=config_p, ch=config.detect.channels, nc=nc).to(self.device)
	        logging.info(f'init {model_t} with (nc: {nc})')
	        self.net = net
	
	        # init hyperparameters
	        hyp = config.loss.detect
	        nl = net.model[-1].nl  # number of detection layers
	
	        # model parameters
	        hyp['box'] *= 3 / nl  # scale to layers
	        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
	        hyp['obj'] *= (config.train.image_size[0] / 640) ** 2 * 3 / nl  # scale to image size and layers
	        hyp['label_smoothing'] = False  # label smoothing
	
	        # attach constants
	        net.nc = nc  # attach number of classes to model
	        net.hyp = hyp  # attach hyper parameters to model
	        net.class_weights = labels_to_class_weights(labels, nc).to(self.device)  # attach class weights
	        net.names = classes
	
	        # load pretrained parameters (optional)
	        d_ckpt = config.detect.pretrained
	        if d_ckpt is not None:
	            if 'http' in d_ckpt:
	                ckpt_p = Path.cwd() / 'weights' / 'v1' / 'tardal.pth'
	                url = d_ckpt
	                logging.info(f'download pretrained parameters from {url}')
	                try:
	                    ckpt = torch.hub.load_state_dict_from_url(url, model_dir=ckpt_p.parent, map_location='cpu')
	                except Exception as err:
	                    logging.fatal(f'connect to {url} failed: {err}, try download pretrained weights manually')
	                    sys.exit(1)
	            else:
	                ckpt = torch.load(d_ckpt, map_location='cpu')
	            # self.load_ckpt(ckpt)
	
	        # criterion (reference: YOLOv5 official)
	        self.loss = ComputeLoss(net)

其中

	self.load_ckpt(ckpt)

注释掉

###然后看log就行

##实验结果

原文：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1739850251806.png)

改为我们的pipeline：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/8e52b1b38028d06431a31758822f4e6.png)