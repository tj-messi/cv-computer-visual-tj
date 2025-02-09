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

其中model.load网上的模型不能成功需要下载下来到本地使用

参考csdn上的本地模型解决方法

	# MODEL_NAME = "meta-llama/Llama-2-7b-hf"
	MODEL_NAME = "./Llama-2-7b-chat-hf"

	def create_model_and_tokenizer():
	    bnb_config = BitsAndBytesConfig(
	        load_in_4bit=True,
	        bnb_4bit_quant_type="nf4",
	        bnb_4bit_compute_dtype=torch.float16,
	    )
	
	    model = AutoModelForCausalLM.from_pretrained(
	        MODEL_NAME,
	        use_safetensors=True,
	        quantization_config=bnb_config,
	        trust_remote_code=True,
	        device_map="auto",
	    )
	
	    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	    tokenizer.pad_token = tokenizer.eos_token
	    tokenizer.padding_side = "right"
	
	    return model, tokenizer

把dataloaderGPT2Classification.py中108行

	self.image_processor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

改为现在模型的相对路径
	
	self.image_processor = AutoFeatureExtractor.from_pretrained("./model_pt_dir")

这个相对路径里面保存着模型

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1738939317937.png)

gpt-2也需要如法炮制



##创新点复现