# ICA-Fusion

##代码

git下来上传，环境配置没问题

上传一个yolov5的pt预训练模型

记得看一下np.int已经被弃用了

##数据



##禁止load pretrain

    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        new_state_dict = state_dict
        for key in list(state_dict.keys()):
            new_state_dict[key[:6] + str(int(key[6])+10) + key[7:]] = state_dict[key]
        # model.load_state_dict(new_state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

其中

	 # model.load_state_dict(new_state_dict, strict=False)  # load

注释掉


##实验结果

###他的


###我们的pipeline