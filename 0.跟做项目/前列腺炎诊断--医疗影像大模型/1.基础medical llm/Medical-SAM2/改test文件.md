#改test文件

train_2d.py 文件

	# train.py
	#!/usr/bin/env	python3
	
	""" train network using pytorch
	    Jiayuan Zhu
	"""
	
	import os
	import time
	
	import torch
	import torch.optim as optim
	import torchvision.transforms as transforms
	from tensorboardX import SummaryWriter
	#from dataset import *
	from torch.utils.data import DataLoader
	
	import cfg
	import func_2d.function as function
	from conf import settings
	#from models.discriminatorlayer import discriminator
	from func_2d.dataset import *
	from func_2d.utils import *
	
	
	def main():
	    # use bfloat16 for the entire work
	    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
	
	    if torch.cuda.get_device_properties(0).major >= 8:
	        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
	        torch.backends.cuda.matmul.allow_tf32 = True
	        torch.backends.cudnn.allow_tf32 = True
	
	
	    args = cfg.parse_args()
	    GPUdevice = torch.device('cuda', args.gpu_device)
	
	    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
	
	    # optimisation
	    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 
	
	    '''load pretrained model'''
	
	    args.path_helper = set_log_dir('logs', args.exp_name)
	    logger = create_logger(args.path_helper['log_path'])
	    logger.info(args)
	
	
	    '''segmentation data'''
	    transform_train = transforms.Compose([
	        transforms.Resize((args.image_size,args.image_size)),
	        transforms.ToTensor(),
	    ])
	
	    transform_test = transforms.Compose([
	        transforms.Resize((args.image_size, args.image_size)),
	        transforms.ToTensor(),
	    ])
	
	    
	    # example of REFUGE dataset
	    if args.dataset == 'REFUGE':
	        '''REFUGE data'''
	        refuge_train_dataset = REFUGE(args, args.data_path, transform = transform_train, mode = 'Training')
	        refuge_test_dataset = REFUGE(args, args.data_path, transform = transform_test, mode = 'Test')
	
	        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
	        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
	        '''end'''
	
	    if args.dataset == 'US_PROSTATE':
	        '''us-prostate data'''
	        us_prostate_train_dataset = US_PROSTATE(args, args.data_path, transform = transform_train, mode = 'Training')
	        us_prostate_test_dataset = US_PROSTATE(args, args.data_path, transform = transform_test, mode = 'Test')
	
	        nice_train_loader = DataLoader(us_prostate_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
	        nice_test_loader = DataLoader(us_prostate_test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
	
	
	
	    '''checkpoint path and tensorboard'''
	    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
	    #use tensorboard
	    if not os.path.exists(settings.LOG_DIR):
	        os.mkdir(settings.LOG_DIR)
	    writer = SummaryWriter(log_dir=os.path.join(
	            settings.LOG_DIR, args.net, settings.TIME_NOW))
	
	    #create checkpoint folder to save model
	    if not os.path.exists(checkpoint_path):
	        os.makedirs(checkpoint_path)
	    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
	
	
	    '''begain training'''
	    best_tol = 1e4
	    best_dice = 0.0
	
	
	    for epoch in range(settings.EPOCH):
	
	        if epoch == 0:
	            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
	            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
	
	        # training
	        net.train()
	        time_start = time.time()
	        loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer)
	        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
	        time_end = time.time()
	        print('time_for_training ', time_end - time_start)
	
	        # validation
	        net.eval()
	        if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
	
	            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
	            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
	
	            if edice > best_dice:
	                best_dice = edice
	                torch.save({'model': net.state_dict(), 'parameter': net._parameters}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))
	
	
	    writer.close()
	
	
	if __name__ == '__main__':
	    main()

vis-image函数文件

	def vis_image(imgs, pred_masks, gt_masks, save_path, reverse = False, points = None):
	    
	    b,c,h,w = pred_masks.size()
	    dev = pred_masks.get_device()
	    row_num = min(b, 4)
	
	    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
	        pred_masks = torch.sigmoid(pred_masks)
	
	    if reverse == True:
	        pred_masks = 1 - pred_masks
	        gt_masks = 1 - gt_masks
	    if c == 2: # for REFUGE multi mask output
	        pred_disc, pred_cup = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), pred_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
	        gt_disc, gt_cup = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), gt_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
	        tup = (imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:])
	        compose = torch.cat(tup, 0)
	        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
	    elif c > 2: # for multi-class segmentation > 2 classes
	        preds = []
	        gts = []
	        for i in range(0, c):
	            pred = pred_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
	            preds.append(pred)
	            gt = gt_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
	            gts.append(gt)
	        tup = [imgs[:row_num,:,:,:]] + preds + gts
	        compose = torch.cat(tup,0)
	        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
	    else:
	        imgs = torchvision.transforms.Resize((h,w))(imgs)
	        if imgs.size(1) == 1:
	            imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
	        pred_masks = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
	        gt_masks = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
	        if points != None:
	            for i in range(b):
	
	                p = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
	                
	                gt_masks[i,0,p[i,0]-2:p[i,0]+2,p[i,1]-2:p[i,1]+2] = 0.5
	                gt_masks[i,1,p[i,0]-2:p[i,0]+2,p[i,1]-2:p[i,1]+2] = 0.1
	                gt_masks[i,2,p[i,0]-2:p[i,0]+2,p[i,1]-2:p[i,1]+2] = 0.4
	                # gt_masks[i,0,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.5
	                # gt_masks[i,1,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.1
	                # gt_masks[i,2,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.4
	        tup = (imgs[:row_num,:,:,:],pred_masks[:row_num,:,:,:], gt_masks[:row_num,:,:,:])
	        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
	        compose = torch.cat(tup,0)
	        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
	
	    return

validation_sam 函数

	def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
	
	    # use bfloat16 for the entire notebook
	    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
	
	    if torch.cuda.get_device_properties(0).major >= 8:
	        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
	        torch.backends.cuda.matmul.allow_tf32 = True
	        torch.backends.cudnn.allow_tf32 = True
	
	
	    # eval mode
	    net.eval()
	
	    n_val = len(val_loader) 
	    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
	    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
	
	    # init
	    lossfunc = criterion_G
	    memory_bank_list = []
	    feat_sizes = [(256, 256), (128, 128), (64, 64)]
	    total_loss = 0
	    total_eiou = 0
	    total_dice = 0
	
	
	    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
	        for ind, pack in enumerate(val_loader):
	            to_cat_memory = []
	            to_cat_memory_pos = []
	            to_cat_image_embed = []
	
	            name = pack['image_meta_dict']['filename_or_obj']
	            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
	            masks = pack['mask'].to(dtype = torch.float32, device = GPUdevice)
	
	            
	            if 'pt' in pack:
	                pt_temp = pack['pt'].to(device = GPUdevice)
	                pt = pt_temp.unsqueeze(1)
	                point_labels_temp = pack['p_label'].to(device = GPUdevice)
	                point_labels = point_labels_temp.unsqueeze(1)
	                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
	                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
	            else:
	                coords_torch = None
	                labels_torch = None
	
	
	
	            '''test'''
	            with torch.no_grad():
	
	                """ image encoder """
	                backbone_out = net.forward_image(imgs)
	                _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
	                B = vision_feats[-1].size(1) 
	
	                """ memory condition """
	                if len(memory_bank_list) == 0:
	                    vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
	                    vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
	
	                else:
	                    for element in memory_bank_list:
	                        maskmem_features = element[0]
	                        maskmem_pos_enc = element[1]
	                        to_cat_memory.append(maskmem_features.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
	                        to_cat_memory_pos.append(maskmem_pos_enc.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
	                        to_cat_image_embed.append((element[3]).cuda(non_blocking=True)) # image_embed
	                        
	                    memory_stack_ori = torch.stack(to_cat_memory, dim=0)
	                    memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
	                    image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)
	
	                    vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64) 
	                    vision_feats_temp = vision_feats_temp.reshape(B, -1)
	
	                    image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
	                    vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
	                    similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
	
	                    similarity_scores = F.softmax(similarity_scores, dim=1) 
	                    sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)  # Shape [batch_size, 16]
	
	                    memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
	                    memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))
	
	                    memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
	                    memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))
	
	
	
	                    vision_feats[-1] = net.memory_attention(
	                        curr=[vision_feats[-1]],
	                        curr_pos=[vision_pos_embeds[-1]],
	                        memory=memory,
	                        memory_pos=memory_pos,
	                        num_obj_ptr_tokens=0
	                        )
	
	                feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) 
	                        for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
	                
	                image_embed = feats[-1]
	                high_res_feats = feats[:-1]
	
	                """ prompt encoder """
	                if (ind%5) == 0:
	                    flag = True
	                    points = (coords_torch, labels_torch)
	
	                else:
	                    flag = False
	                    points = None
	
	                se, de = net.sam_prompt_encoder(
	                    points=points, 
	                    boxes=None,
	                    masks=None,
	                    batch_size=B,
	                )
	
	                low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
	                    image_embeddings=image_embed,
	                    image_pe=net.sam_prompt_encoder.get_dense_pe(), 
	                    sparse_prompt_embeddings=se,
	                    dense_prompt_embeddings=de, 
	                    multimask_output=False, 
	                    repeat_image=False,  
	                    high_res_features = high_res_feats
	                )
	
	                # prediction
	                pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size))
	                high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
	                                                mode="bilinear", align_corners=False)
	            
	                """ memory encoder """
	                maskmem_features, maskmem_pos_enc = net._encode_new_memory( 
	                    current_vision_feats=vision_feats,
	                    feat_sizes=feat_sizes,
	                    pred_masks_high_res=high_res_multimasks,
	                    is_mask_from_pts=flag)  
	                    
	                maskmem_features = maskmem_features.to(torch.bfloat16)
	                maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
	                maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
	                maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)
	
	
	                """ memory bank """
	                if len(memory_bank_list) < 16:
	                    for batch in range(maskmem_features.size(0)):
	                        memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
	                                                 (maskmem_pos_enc[batch].unsqueeze(0)),
	                                                 iou_predictions[batch, 0],
	                                                 image_embed[batch].reshape(-1).detach()])
	                
	                else:
	                    for batch in range(maskmem_features.size(0)):
	                        
	                        memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
	                        memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)
	
	                        memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
	                        current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
	                                                             memory_bank_maskmem_features_norm.t())
	
	                        current_similarity_matrix_no_diag = current_similarity_matrix.clone()
	                        diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
	                        current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')
	
	                        single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
	                        similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
	                        min_similarity_index = torch.argmin(similarity_scores) 
	                        max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])
	
	                        if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
	                            if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
	                                memory_bank_list.pop(max_similarity_index) 
	                                memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
	                                                         (maskmem_pos_enc[batch].unsqueeze(0)),
	                                                         iou_predictions[batch, 0],
	                                                         image_embed[batch].reshape(-1).detach()])
	
	                # binary mask and calculate loss, iou, dice
	                total_loss += lossfunc(pred, masks)
	                pred = (pred> 0.5).float()
	                temp = eval_seg(pred, masks, threshold)
	                total_eiou += temp[0]
	                total_dice += temp[1]
	
	                '''vis images'''
	                if ind % args.vis == 0:
	                    namecat = 'Test'
	                    for na in name:
	                        img_name = na
	                        namecat = namecat + img_name + '+'
	                    vis_image(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=None)
	                            
	            pbar.update()
	
	    return total_loss/ n_val , tuple([total_eiou/n_val, total_dice/n_val])
	
###最后效果

	import os
	import time
	
	import torch.nn.functional as F
	from tqdm import tqdm
	import torch
	import torch.optim as optim
	import torchvision.transforms as transforms
	from tensorboardX import SummaryWriter
	#from dataset import *
	from torch.utils.data import DataLoader
	
	import cfg
	import func_2d.function as function
	from conf import settings
	#from models.discriminatorlayer import discriminator
	from func_2d.dataset import *
	from func_2d.utils import *
	
	def vis_image_for_input(imgs, pred_masks, gt_masks, save_path, reverse = False, points = None):
	    
	    b,c,h,w = pred_masks.size()
	    dev = pred_masks.get_device()
	    row_num = min(b, 4)
	
	    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
	        pred_masks = torch.sigmoid(pred_masks)
	
	    if reverse == True:
	        pred_masks = 1 - pred_masks
	        gt_masks = 1 - gt_masks
	    if c == 2: # for REFUGE multi mask output
	        pred_disc, pred_cup = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), pred_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
	        gt_disc, gt_cup = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), gt_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
	        tup = (imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:])
	        compose = torch.cat(tup, 0)
	        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
	    elif c > 2: # for multi-class segmentation > 2 classes
	        preds = []
	        gts = []
	        for i in range(0, c):
	            pred = pred_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
	            preds.append(pred)
	            gt = gt_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
	            gts.append(gt)
	        tup = [imgs[:row_num,:,:,:]] + preds + gts
	        compose = torch.cat(tup,0)
	        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
	    else:
	        #二分割分类任务
	        #print("test-binary-segment\n")
	        imgs = torchvision.transforms.Resize((h,w))(imgs)
	        if imgs.size(1) == 1:
	            imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
	        pred_masks = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
	        gt_masks = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
	        if points != None:
	            for i in range(b):
	
	                p = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
	                
	                gt_masks[i,0,p[i,0]-2:p[i,0]+2,p[i,1]-2:p[i,1]+2] = 0.5
	                gt_masks[i,1,p[i,0]-2:p[i,0]+2,p[i,1]-2:p[i,1]+2] = 0.1
	                gt_masks[i,2,p[i,0]-2:p[i,0]+2,p[i,1]-2:p[i,1]+2] = 0.4
	                # gt_masks[i,0,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.5
	                # gt_masks[i,1,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.1
	                # gt_masks[i,2,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.4
	        tup = (imgs[:row_num,:,:,:],pred_masks[:row_num,:,:,:], gt_masks[:row_num,:,:,:])
	        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
	        compose = torch.cat(tup,0)
	        
	        # 保存每张分割结果
	        for i in range(row_num):
	            single_pred_mask = pred_masks[i, :, :, :].unsqueeze(0)  # 获取单张预测掩码
	            single_save_path = save_path.replace(".jpg", f"_segment_{i}.png")  # 为每张图片生成唯一文件名
	            vutils.save_image(single_pred_mask, fp=single_save_path, nrow=1, padding=10)  # 保存单张分割结果
	        
	        #保存合并可视化内容
	        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
	
	    return
	
	def validation_for_input(args, val_loader, epoch, net: nn.Module, clean_dir=True):
	     # use bfloat16 for the entire notebook
	    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
	
	    if torch.cuda.get_device_properties(0).major >= 8:
	        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
	        torch.backends.cuda.matmul.allow_tf32 = True
	        torch.backends.cudnn.allow_tf32 = True
	
	
	    # eval mode
	    net.eval()
	
	    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
	    pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
	    criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
	
	    n_val = len(val_loader) 
	    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
	
	
	    # init
	    lossfunc = criterion_G
	    memory_bank_list = []
	    feat_sizes = [(256, 256), (128, 128), (64, 64)] #特征图输入
	    total_loss = 0
	    total_eiou = 0
	    total_dice = 0
	
	    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
	        for ind, pack in enumerate(val_loader):
	            to_cat_memory = []
	            to_cat_memory_pos = []
	            to_cat_image_embed = []
	
	            name = pack['image_meta_dict']['filename_or_obj']
	            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
	            masks = pack['mask'].to(dtype = torch.float32, device = GPUdevice)
	
	            
	            if 'pt' in pack:
	                pt_temp = pack['pt'].to(device = GPUdevice)
	                pt = pt_temp.unsqueeze(1)
	                point_labels_temp = pack['p_label'].to(device = GPUdevice)
	                point_labels = point_labels_temp.unsqueeze(1)
	                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
	                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
	            else:
	                coords_torch = None
	                labels_torch = None
	
	
	
	            '''test'''
	            with torch.no_grad():
	
	                """ image encoder """
	                backbone_out = net.forward_image(imgs)
	                _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
	                B = vision_feats[-1].size(1) 
	
	                """ memory condition """
	                if len(memory_bank_list) == 0:
	                    vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
	                    vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
	
	                else:
	                    for element in memory_bank_list:
	                        maskmem_features = element[0]
	                        maskmem_pos_enc = element[1]
	                        to_cat_memory.append(maskmem_features.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
	                        to_cat_memory_pos.append(maskmem_pos_enc.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
	                        to_cat_image_embed.append((element[3]).cuda(non_blocking=True)) # image_embed
	                        
	                    memory_stack_ori = torch.stack(to_cat_memory, dim=0)
	                    memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
	                    image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)
	
	                    vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64) 
	                    vision_feats_temp = vision_feats_temp.reshape(B, -1)
	
	                    image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
	                    vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
	                    similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
	
	                    similarity_scores = F.softmax(similarity_scores, dim=1) 
	                    sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)  # Shape [batch_size, 16]
	
	                    memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
	                    memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))
	
	                    memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
	                    memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))
	
	
	
	                    vision_feats[-1] = net.memory_attention(
	                        curr=[vision_feats[-1]],
	                        curr_pos=[vision_pos_embeds[-1]],
	                        memory=memory,
	                        memory_pos=memory_pos,
	                        num_obj_ptr_tokens=0
	                        )
	
	                feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) 
	                        for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
	                
	                image_embed = feats[-1]
	                high_res_feats = feats[:-1]
	
	                """ prompt encoder """
	                if (ind%5) == 0:
	                    flag = True
	                    points = (coords_torch, labels_torch)
	
	                else:
	                    flag = False
	                    points = None
	
	                se, de = net.sam_prompt_encoder(
	                    points=points, 
	                    boxes=None,
	                    masks=None,
	                    batch_size=B,
	                )
	
	                low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
	                    image_embeddings=image_embed,
	                    image_pe=net.sam_prompt_encoder.get_dense_pe(), 
	                    sparse_prompt_embeddings=se,
	                    dense_prompt_embeddings=de, 
	                    multimask_output=False, 
	                    repeat_image=False,  
	                    high_res_features = high_res_feats
	                )
	
	                # prediction
	                pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size))
	                high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
	                                                mode="bilinear", align_corners=False)
	            
	                """ memory encoder """
	                maskmem_features, maskmem_pos_enc = net._encode_new_memory( 
	                    current_vision_feats=vision_feats,
	                    feat_sizes=feat_sizes,
	                    pred_masks_high_res=high_res_multimasks,
	                    is_mask_from_pts=flag)  
	                    
	                maskmem_features = maskmem_features.to(torch.bfloat16)
	                maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
	                maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
	                maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)
	
	
	                """ memory bank """
	                if len(memory_bank_list) < 16:
	                    for batch in range(maskmem_features.size(0)):
	                        memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
	                                                 (maskmem_pos_enc[batch].unsqueeze(0)),
	                                                 iou_predictions[batch, 0],
	                                                 image_embed[batch].reshape(-1).detach()])
	                
	                else:
	                    for batch in range(maskmem_features.size(0)):
	                        
	                        memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
	                        memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)
	
	                        memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
	                        current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
	                                                             memory_bank_maskmem_features_norm.t())
	
	                        current_similarity_matrix_no_diag = current_similarity_matrix.clone()
	                        diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
	                        current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')
	
	                        single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
	                        similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
	                        min_similarity_index = torch.argmin(similarity_scores) 
	                        max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])
	
	                        if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
	                            if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
	                                memory_bank_list.pop(max_similarity_index) 
	                                memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
	                                                         (maskmem_pos_enc[batch].unsqueeze(0)),
	                                                         iou_predictions[batch, 0],
	                                                         image_embed[batch].reshape(-1).detach()])
	
	                # binary mask and calculate loss, iou, dice
	                total_loss += lossfunc(pred, masks)
	                pred = (pred> 0.5).float()
	                temp = eval_seg(pred, masks, threshold)
	                total_eiou += temp[0]
	                total_dice += temp[1]
	
	                '''vis images'''
	                if ind % args.vis == 0:
	                    namecat = 'Test'
	                    for na in name:
	                        img_name = na
	                        namecat = namecat + img_name + '+'
	                    vis_image_for_input(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=None)
	                            
	            pbar.update()
	
	    return total_loss/ n_val , tuple([total_eiou/n_val, total_dice/n_val])
	    
	
	
	def main():
	     # use bfloat16 for the entire work
	    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
	
	    if torch.cuda.get_device_properties(0).major >= 8:
	        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
	        torch.backends.cuda.matmul.allow_tf32 = True
	        torch.backends.cudnn.allow_tf32 = True
	
	
	    args = cfg.parse_args()
	    GPUdevice = torch.device('cuda', args.gpu_device)
	
	    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
	
	     # optimisation
	    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 
	
	    '''load pretrained model'''
	
	    args.path_helper = set_log_dir('logs', args.exp_name)
	    logger = create_logger(args.path_helper['log_path'])
	    logger.info(args)
	
	    '''segmentation data'''
	    transform_train = transforms.Compose([
	        transforms.Resize((args.image_size,args.image_size)),
	        transforms.ToTensor(),
	    ])
	
	    transform_test = transforms.Compose([
	        transforms.Resize((args.image_size, args.image_size)),
	        transforms.ToTensor(),
	    ])
	
	    '''checkpoint path and tensorboard'''
	    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
	    #use tensorboard
	    if not os.path.exists(settings.LOG_DIR):
	        os.mkdir(settings.LOG_DIR)
	    writer = SummaryWriter(log_dir=os.path.join(
	            settings.LOG_DIR, args.net, settings.TIME_NOW))
	
	    #create checkpoint folder to save model
	    if not os.path.exists(checkpoint_path):
	        os.makedirs(checkpoint_path)
	    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
	
	    #validation
	    net.eval()
	
	    epoch = 1
	    us_prostate_test_dataset = US_PROSTATE(args, args.data_path, transform = transform_test, mode = 'Test')
	    nice_test_loader = DataLoader(us_prostate_test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
	
	    tol, (eiou, edice) = validation_for_input(args, nice_test_loader, epoch, net, writer)
	
	
	if __name__ == "__main__":
	    main()
