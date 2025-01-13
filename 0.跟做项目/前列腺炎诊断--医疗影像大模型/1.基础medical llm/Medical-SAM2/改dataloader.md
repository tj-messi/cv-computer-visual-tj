#修改dataloader

原代码格式

    if args.dataset == 'REFUGE':
        '''REFUGE data'''
        refuge_train_dataset = REFUGE(args, args.data_path, transform = transform_train, mode = 'Training')
        refuge_test_dataset = REFUGE(args, args.data_path, transform = transform_test, mode = 'Test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
        '''end'''

其中的REFUGE调用如下：

	class REFUGE(Dataset):
	    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):
	        self.data_path = data_path
	        self.subfolders = [f.path for f in os.scandir(os.path.join(data_path, mode + '-400')) if f.is_dir()]
	        self.mode = mode
	        self.prompt = prompt
	        self.img_size = args.image_size
	        self.mask_size = args.out_size
	
	        self.transform = transform
	        self.transform_msk = transform_msk
	
	    def __len__(self):
	        return len(self.subfolders)
	
	    def __getitem__(self, index):
	
	        """Get the images"""
	        subfolder = self.subfolders[index]
	        name = subfolder.split('/')[-1]
	
	        # raw image and raters path
	        img_path = os.path.join(subfolder, name + '_cropped.jpg')
	        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '_cropped.jpg') for i in range(1, 8)]
	
	        # img_path = os.path.join(subfolder, name + '.jpg')
	        # multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '.png') for i in range(1, 8)]
	
	        # raw image and rater images
	        img = Image.open(img_path).convert('RGB')
	        multi_rater_cup = [Image.open(path).convert('L') for path in multi_rater_cup_path]
	
	        # apply transform
	        if self.transform:
	            state = torch.get_rng_state()
	            img = self.transform(img)
	            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >=0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup]
	            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)
	
	            torch.set_rng_state(state)
	
	        # find init click and apply majority vote
	        if self.prompt == 'click':
	
	            point_label_cup, pt_cup = random_click(np.array((multi_rater_cup.mean(axis=0)).squeeze(0)), point_label = 1)
	            
	            selected_rater_mask_cup_ori = multi_rater_cup.mean(axis=0)
	            selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float() 
	
	
	            selected_rater_mask_cup = F.interpolate(selected_rater_mask_cup_ori.unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0) # torch.Size([1, mask_size, mask_size])
	            selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()
	
	
	            # # Or use any specific rater as GT
	            # point_label_cup, pt_cup = random_click(np.array(multi_rater_cup[0, :, :, :].squeeze(0)), point_label = 1)
	            # selected_rater_mask_cup_ori = multi_rater_cup[0,:,:,:]
	            # selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float() 
	
	            # selected_rater_mask_cup = F.interpolate(selected_rater_mask_cup_ori.unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0) # torch.Size([1, mask_size, mask_size])
	            # selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()
	
	
	        image_meta_dict = {'filename_or_obj':name}
	        return {
	            'image':img,
	            'multi_rater': multi_rater_cup, 
	            'p_label': point_label_cup,
	            'pt':pt_cup, 
	            'mask': selected_rater_mask_cup, 
	            'mask_ori': selected_rater_mask_cup_ori,
	            'image_meta_dict':image_meta_dict,
	        }
	

改成US-PROSTATE的内容就是

	class US_PROSTATE(Dataset):
	    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):
	        self.data_path = data_path
	        self.subfolders = [f.path for f in os.scandir(os.path.join(data_path, mode)) if f.is_dir()] #检查大文件夹
	        self.mode = mode
	        self.prompt = prompt
	        self.img_size = args.image_size
	        self.mask_size = args.out_size
	
	        self.transform = transform
	        self.transform_msk = transform_msk
	
	    def __len__(self):
	        return len(self.subfolders)
	    
	    def __getitem__(self, index):
	
	        """Get the images"""
	        subfolder = self.subfolders[index] #单个子文件夹
	        name = subfolder.split('/')[-1] #单个子文件夹名字
	
	         # raw image and raters path
	        img_path = os.path.join(subfolder, name + '_img.png')
	        label_path = os.path.join(subfolder, name + '_label.png')
	
	        # raw image and rater images
	        img = Image.open(img_path).convert('RGB')
	        label = [Image.open(label_path).convert('L') ]
	
	        # 对图像和掩码应用变换
	        if self.transform:
	            state = torch.get_rng_state()  # 保存随机状态
	            img = self.transform(img)
	            label = [torch.as_tensor((self.transform(single_label) >= 0.5).float(), dtype=torch.float32) for single_label in label]
	            label = torch.stack(label, dim=0)  # 将所有标注者的掩码堆叠为一个张量
	            torch.set_rng_state(state)  # 恢复随机状态
	
	         # 根据提示类型生成初始点击点
	        if self.prompt == 'click':
	            point_label, pt = random_click(np.array((label.mean(axis=0)).squeeze(0)), point_label=1)
	            
	            # 计算多标注者的平均掩码，并二值化
	            selected_rater_mask_ori = label.mean(axis=0)
	            selected_rater_mask_ori = (selected_rater_mask_ori >= 0.5).float()
	
	            # 调整掩码大小到指定输出大小
	            selected_rater_mask = F.interpolate(
	                selected_rater_mask_ori.unsqueeze(0),
	                size=(self.mask_size, self.mask_size),
	                mode='bilinear',
	                align_corners=False
	            ).mean(dim=0)
	            selected_rater_mask = (selected_rater_mask >= 0.5).float()
	
	        # 图像元信息字典
	        image_meta_dict = {'filename_or_obj': name}
	
	        return {
	            'image': img,
	            'multi_rater': label,
	            'p_label': point_label,
	            'pt': pt,
	            'mask': selected_rater_mask,
	            'mask_ori': selected_rater_mask_ori,
	            'image_meta_dict': image_meta_dict,
	        }
	        '''
	        return
	        {
	            'image':img,
	            'multi_rater': multi_rater_cup, 
	            'p_label': point_label_cup,
	            'pt':pt_cup, 
	            'mask': selected_rater_mask_cup, 
	            'mask_ori': selected_rater_mask_cup_ori,
	            'image_meta_dict':image_meta_dict,
	        }
	      

最后

	python train_2d.py -net sam2 -exp_name US_PROSTATE_MedSAM2 -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset US_PROSTATE -data_path ./data/US_PROSTATE