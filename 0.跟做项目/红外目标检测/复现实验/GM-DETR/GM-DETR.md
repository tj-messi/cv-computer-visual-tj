#GM-DETR

##代码git

没问题

##数据集整理

注意一下数据集整理的格式，需要的数据集格式是coco，我们拥有的数据集格式是yolo

转成coco数据集格式即可

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1739898125726.png)

##禁止pre_train

###PResNetX
	
	class PResNetX(nn.Module):
	    def __init__(
	            self,
	            depth,
	            variant='d',
	            num_stages=4,
	            return_idx=[0, 1, 2, 3],
	            act='relu',
	            freeze_at=-1,
	            freeze_norm=True,
	            pretrained=False):
	        super().__init__()
	
	        block_nums = ResNet_cfg[depth]
	        ch_in = 64
	        if variant in ['c', 'd']:
	            conv_def = [
	                [3, ch_in // 2, 3, 2, "conv1_1"],
	                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
	                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
	            ]
	        else:
	            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]
	
	        self.conv1 = nn.Sequential(OrderedDict([
	            (_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def
	        ]))
	
	        ch_out_list = [64, 128, 256, 512]
	        block = BottleNeck if depth >= 50 else BasicBlock
	
	        _out_channels = [block.expansion * v for v in ch_out_list]
	        _out_strides = [4, 8, 16, 32]
	
	        self.res_layers = nn.ModuleList()
	        for i in range(num_stages):
	            stage_num = i + 2
	            self.res_layers.append(
	                Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant)
	            )
	            ch_in = _out_channels[i]
	
	        self.return_idx = return_idx
	        self.out_channels = [_out_channels[_i] for _i in return_idx]
	        self.out_strides = [_out_strides[_i] for _i in return_idx]
	
	        if freeze_at >= 0:
	            self._freeze_parameters(self.conv1)
	            for i in range(min(freeze_at, num_stages)):
	                self._freeze_parameters(self.res_layers[i])
	
	        if freeze_norm:
	            self._freeze_norm(self)
	
	        if pretrained:
	            state = torch.hub.load_state_dict_from_url(donwload_url[depth])
	            # self.load_state_dict(state)
	            print(f'Load PResNet{depth} state_dict')

其中

	self.load_state_dict(state)加载

直接注释掉

##实验结果

###原论文的结果

没有这个数据集

###我们的结果