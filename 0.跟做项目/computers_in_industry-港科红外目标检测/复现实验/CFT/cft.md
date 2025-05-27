# CFT

##代码

直接git

##数据

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1739962443891.png)

##禁止load pretrain


	    # Model
	    pretrained = weights.endswith('.pt')
	    if pretrained:
	        with torch_distributed_zero_first(rank):
	            attempt_download(weights)  # download if not found locally
	        ckpt = torch.load(weights, map_location=device)  # load checkpoint
	        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
	        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
	        state_dict = ckpt['model'].float().state_dict()  # to FP32
	        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
	        # model.load_state_dict(state_dict, strict=False)  # load
	        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
	    else:
	        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
	    with torch_distributed_zero_first(rank):
	        check_dataset(data_dict)  # check
	    train_path = data_dict['train']
	    test_path = data_dict['val']

修改model的load就行了