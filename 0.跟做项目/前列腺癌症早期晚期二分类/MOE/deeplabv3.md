#deeplabv3

##代码

两个代码train和dataset

	import os
	import torch
	from torch.utils.data import DataLoader
	from torchvision import models, transforms
	import torch.nn as nn
	from tqdm import tqdm
	from PIL import Image
	import numpy as np
	
	from dataset import *
	
	def dice_score(pred, target, threshold=0.5):
	    # 归一化输出，将值转换为 0 或 1
	    pred = (pred > threshold).astype(np.uint8)
	    target = (target > threshold).astype(np.uint8)
	    
	    intersection = np.sum(pred * target)
	    return 2. * intersection / (np.sum(pred) + np.sum(target) + 1e-6)  # 防止除零错误
	
	def save_combined_mask(gt_mask, predicted_mask, file_path):
	    gt_mask = np.squeeze(gt_mask)  # 将 GT 掩码转换为 2D 数组
	    predicted_mask = np.squeeze(predicted_mask)  # 确保预测掩码是 2D 数组
	
	
	    # 合并 GT 掩码和预测掩码
	    combined_mask = np.concatenate((gt_mask, predicted_mask), axis=1)  # 水平拼接
	    combined_mask_image = Image.fromarray(combined_mask.astype(np.uint8) * 255)  # 转为黑白图像
	    combined_mask_image.save(file_path)
	    combined_mask_image.save(file_path)
	
	transform = transforms.Compose([
	    transforms.Resize((256, 256)),
	    transforms.ToTensor(),
	])
	
	# 数据加载器
	train_dataset = MedicalSegmentationDataset(image_dir='/media/tongji/Deeplabv3/data/training/input', mask_dir='/media/tongji/Deeplabv3/data/training/output', transform=transform)
	train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
	
	val_dataset = MedicalSegmentationDataset(image_dir='/media/tongji/Deeplabv3/data/val/input', mask_dir='/media/tongji/Deeplabv3/data/val/output', transform=transform)
	val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
	
	# 加载预训练的 DeepLabV3 模型
	model = models.segmentation.deeplabv3_resnet101(pretrained=True)
	
	# 修改最后的分类层，假设我们有两个类别：背景和病灶
	num_classes = 2  # 1个病灶类别 + 1个背景类别
	in_channels = model.classifier[4].in_channels
	model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1))
	
	# 将模型转到 GPU（如果可用）
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	
	# 损失函数和优化器
	criterion = nn.CrossEntropyLoss()  # 适用于二分类问题
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	
	num_epochs = 10
	best_dice_score = 0.0
	for epoch in range(num_epochs):
	    model.train()
	    running_loss = 0.0
	    # 使用 tqdm 显示进度条
	    for images, masks, img_names in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
	        images, masks = images.to(device), masks.to(device)
	
	        optimizer.zero_grad()
	
	        # 前向传播
	        outputs = model(images)["out"]  # 获取模型输出
	
	        masks = masks.squeeze(1)
	        masks = masks.long()
	
	        # 计算损失
	        loss = criterion(outputs, masks)
	        loss.backward()
	
	        # 更新权重
	        optimizer.step()
	
	        running_loss += loss.item()
	
	    # 显示当前 epoch 的平均损失
	    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')
	
	    model.eval()
	    dice_scores = []
	    with torch.no_grad():  # 在推理时禁用梯度计算
	        for images, masks, img_names in tqdm(val_loader, desc="Validation", unit="batch"):
	            images, masks = images.to(device), masks.to(device)
	
	            # 进行推理
	            outputs = model(images)["out"]
	            prob = torch.softmax(outputs, dim=1)  # 对于多分类问题
	            print("output:",outputs)
	            print("prob:",prob)
	            predicted_mask = torch.argmax(prob, dim=1).cpu().numpy()  # 选择最大概率的类别作为预测
	
	            # 保存每张图像的掩码
	            if not os.path.exists(f"/media/tongji/Deeplabv3/output/{epoch}"):
	                os.makedirs(f"/media/tongji/Deeplabv3/output/{epoch}")
	            for i in range(predicted_mask.shape[0]):
	                        img_name = img_names[i]  # 获取当前图像的文件名
	                        
	                        # 获取 GT 掩码路径并加载
	                        gt_mask_path = f"/media/tongji/Deeplabv3/data/val/output/{img_name}"
	                        gt_mask = Image.open(gt_mask_path).convert('L')  # 读取为灰度图
	                        gt_mask = transform(gt_mask)
	
	                        # 合并 GT 掩码和预测掩码
	                        save_path = f"/media/tongji/Deeplabv3/output/{epoch}/combined_mask_{img_name}"
	                        save_combined_mask(gt_mask, predicted_mask[i], save_path)
	
	                        # 计算 Dice 系数
	                        score = dice_score(predicted_mask[i], masks[i].cpu().numpy())
	                        dice_scores.append(score)
	
	    
	    # 计算当前 epoch 的平均 Dice 系数
	    avg_dice_score = np.mean(dice_scores)
	    print(f'Average Dice Score for Epoch {epoch+1}: {avg_dice_score}')
	
	    if avg_dice_score > best_dice_score:
	        best_dice_score = avg_dice_score
	        torch.save(model.state_dict(), f'/media/tongji/Deeplabv3/output/best_model.pth')
	        print(f"Saved best model with Dice Score: {best_dice_score}")
	    
	'''    dice_scores = []
	    with torch.no_grad():
	        for images, masks in tqdm(val_loader, desc="Validation", unit="batch"):
	            images, masks = images.to(device), masks.to(device)
	
	            outputs = model(images)["out"]
	            predicted_mask = torch.argmax(outputs, dim=1).cpu().numpy()
	
	            # 计算 Dice 系数
	            for i in range(predicted_mask.shape[0]):
	                score = dice_score(predicted_mask[i], masks[i].cpu().numpy())
	                dice_scores.append(score)
	
	    # 计算平均 Dice 系数
	    print(f'Average Dice Score: {np.mean(dice_scores)}')'''

其中dataset
	
	import os
	import torch
	from torch.utils.data import Dataset, DataLoader
	from torchvision import transforms
	from PIL import Image
	import torch
	import torch.nn as nn
	from torchvision import models
	from torch.utils.data import DataLoader
	from torchvision import transforms
	from PIL import Image
	import os
	
	
	class MedicalSegmentationDataset(torch.utils.data.Dataset):
	    def __init__(self, image_dir, mask_dir, transform=None):
	        self.image_dir = image_dir
	        self.mask_dir = mask_dir
	        self.image_names = os.listdir(image_dir)
	        self.transform = transform
	
	    def __len__(self):
	        return len(self.image_names)
	
	    def __getitem__(self, idx):
	        img_name = self.image_names[idx]
	        img_path = os.path.join(self.image_dir, img_name)
	        mask_path = os.path.join(self.mask_dir, img_name)  # 假设mask和原图同名
	
	        image = Image.open(img_path).convert("RGB")
	        mask = Image.open(mask_path).convert("L")  # 转为灰度图
	
	        if self.transform:
	            image = self.transform(image)
	            mask = self.transform(mask)
	
	        return image, mask,img_name

##数据集

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1740970173634.png)

##train

	python train.py

##test

	import os
	import torch
	from tqdm import tqdm
	from torchvision import models, transforms
	import numpy as np
	from PIL import Image
	from torch.utils.data import DataLoader
	
	from dataset import *
	
	def save_predicted_mask(mask, file_path):
	    """
	    Save the predicted mask as an image.
	    :param mask: The predicted mask (numpy array).
	    :param file_path: The path to save the image.
	    """
	    if mask.ndim == 3:
	        mask = np.argmax(mask,axis=0)
	
	    mask = np.squeeze(mask)  # Remove unnecessary dimensions to ensure 2D (H, W)
	    mask = (mask > 0.5).astype(np.uint8)  # Convert to binary mask (0 or 255)
	
	    print(mask.shape)
	
	    mask_image = Image.fromarray(mask * 255)  # Convert to black and white image (0 for black, 255 for white)
	    mask_image.save(file_path)
	
	transform = transforms.Compose([
	    transforms.Resize((256, 256)),
	    transforms.ToTensor(),
	])
	
	def load_model_with_ignored_keys(model, model_path):
	    # Load the saved model state_dict
	    checkpoint = torch.load(model_path)
	    model_dict = model.state_dict()
	
	    # Create a new state_dict excluding the auxiliary classifier weights
	    checkpoint_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
	
	    # Load the weights
	    model_dict.update(checkpoint_dict)
	    model.load_state_dict(model_dict)
	
	    return model
	
	def test(model, test_loader, device, best_model_path):
	    # 加载保存的最佳模型
	    model = load_model_with_ignored_keys(model, best_model_path)
	    model.to(device)
	    model.eval()  # 切换到评估模式
	
	    with torch.no_grad():  # 禁用梯度计算
	        for images, img_names in tqdm(test_loader, desc="Testing", unit="batch"):
	            images = images.to(device)
	
	            # 进行推理
	            outputs = model(images)["out"]
	            prob = torch.softmax(outputs, dim=1)  # 对于多分类问题
	            predicted_mask = (prob > 0.5).cpu().numpy()  # 选择大于 0.5 的部分作为病灶区域
	
	            # 保存预测掩码
	            for i in range(predicted_mask.shape[0]):
	                img_name = img_names[i]  # 获取当前图像的文件名
	
	                # 保存预测掩码
	                save_path = f"/media/tongji/Deeplabv3/train_output/test/predicted_mask_{img_name}"
	                save_predicted_mask(predicted_mask[i], save_path)  # 保存预测掩码
	
	    print("Test completed!")
	
	# 使用方法
	best_model_path = '/media/tongji/Deeplabv3/train_output/output_3-3-11-0/best_model.pth'
	
	# 创建数据加载器
	test_dataset = MedicalSegmentationDataset(
	    image_dir='/media/tongji/Deeplabv3/data/testing/input', 
	    mask_dir='/media/tongji/Deeplabv3/data/testing/output',  # GT 不是必须的
	    transform=transform,
	    mode = 'test'
	)
	test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
	
	# 创建并初始化模型
	model = models.segmentation.deeplabv3_resnet101(pretrained=False)
	num_classes = 2  # 1个病灶类别 + 1个背景类别
	in_channels = model.classifier[4].in_channels
	model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1))
	
	# 使用 GPU 或 CPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	
	# 运行测试
	test(model, test_loader, device, best_model_path)


with open(job_name,"rb") as job_file :


##直接测试就好

	




