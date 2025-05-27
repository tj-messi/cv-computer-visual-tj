from abc import abstractmethod

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch
import sys
sys.path.append('/home/zeiler/MIA/Diffusion-ST/')
import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pandas as pd
from guided_diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
from guided_diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
class UNet_grid(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        # 解码器
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(48, 16, 3, padding=1),  # 32 = 16(up) + 16(skip)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.up2 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),  # 拼接原始输入
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, 3, padding=1)
        )

    def forward(self, x):
        # 编码
        x1 = self.enc1(x)        # [B,16,H,W]
        x = self.pool1(x1)
        
        x2 = self.enc2(x)        # [B,32,H/2,W/2]
        x = self.pool2(x2)
        
        # 解码
        x = self.up1(x)          # [B,16,H/2,W/2]
        # print(x.shape,x2.shape)
        x = torch.cat([x, x2], dim=1)  # 跳跃连接
        x = self.dec1(x)         # [B,16,H/2,W/2]
        
        x = self.up2(x)          # [B,16,H,W]
        # print(x.shape,x1.shape)
        x = torch.cat([x, x1], dim=1)  # 二次跳跃连接
        x = self.dec2(x)         # [B,out_channels,H,W]
        
        return x
class SimpleRegUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2):
        super().__init__()
        
        # 精简编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),  # [b,16,64,64]
            nn.ReLU(),
            nn.MaxPool2d(2),                     # [b,16,32,32]
            
            nn.Conv2d(16, 32, 3, padding=1),     # [b,32,32,32]
            nn.ReLU(),
            nn.MaxPool2d(2)                      # [b,32,16,16]
        )
        
        # 精简解码器（含跳跃连接）
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),     # [b,64,16,16]
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),          # [b,64,32,32]
            nn.Conv2d(64, 32, 3, padding=1),     # [b,32,32,32]
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),          # [b,32,64,64]
            nn.Conv2d(32, 16, 3, padding=1),     # [b,16,64,64]
            nn.ReLU()
        )
        
        # 形变场预测
        self.flow_pred = nn.Conv2d(16, out_ch, 3, padding=1)

    def forward(self, x):
        # 编码过程
        mid_feat = self.encoder(x)        # 获取中间特征
        
        # 解码过程
        decoded = self.decoder(mid_feat)  # 上采样恢复分辨率
        
        # 预测形变场
        flow = self.flow_pred(decoded)    # [b,2,64,64]
        return flow
class DeformAttention(nn.Module):
    def __init__(self, min_patch=4):
        super().__init__()
        self.min_patch = min_patch
        self.current_patch = torch.tensor(24)  # 当前 patch 大小
        
        # 形变评估网络
        self.deform_net = nn.Sequential(
            nn.Conv2d(128, 512, 3, padding=1),  # 输入通道改为 2（source 和 target 各 1 通道）
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 3, padding=1),  # 输出单通道形变得分
            nn.Sigmoid()  # 形变得分范围 [0, 1]
        )
        
        # 配准图像生成网络
        self.reg_net = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
    def forward(self, source, target):
        # 生成注意力权重图
        attention_map = self._calculate_attention(source, target)
        
        # 生成配准图像
        registered = self.reg_net(torch.cat([source, target], 1))
        
        return registered, attention_map

    def _calculate_attention(self, src, tgt):
        b, c, h, w = src.shape
        patch_size = int(self.current_patch.item())
        stride = patch_size // 2

        # 提取 patches
        patches_src = F.unfold(src, (patch_size, patch_size), stride=stride)  # [b, c*p*p, num_patches]
        patches_tgt = F.unfold(tgt, (patch_size, patch_size), stride=stride)  # [b, c*p*p, num_patches]
        num_patches = patches_src.size(2)

        # 初始化注意力图和计数图
        attention_map = torch.zeros(b, 1, h, w, device=src.device)
        count_map = torch.zeros(b, 1, h, w, device=src.device)

        # 计算每个 patch 的形变得分并累加到注意力图
        for i in range(num_patches):
            src_patch = patches_src[:, :, i].view(b, c, patch_size, patch_size)
            tgt_patch = patches_tgt[:, :, i].view(b, c, patch_size, patch_size)
            patch_pair = torch.cat([src_patch, tgt_patch], 1)  # [b, 2c, p, p]
            score = self.deform_net(patch_pair)  # [b, 1, p, p]

            # 计算 patch 在图像中的位置
            row = (i // ((w - patch_size) // stride + 1)) * stride
            col = (i % ((w - patch_size) // stride + 1)) * stride

            # 累加形变得分到注意力图
            attention_map[:, :, row:row+patch_size, col:col+patch_size] += score
            count_map[:, :, row:row+patch_size, col:col+patch_size] += 1

        # 计算平均形变得分（处理重叠区域）
        attention_map = attention_map / torch.clamp(count_map, min=1)  # 避免除以 0
        
        return attention_map  # [b, 1, h, w]
def compute_gradient(image):
    """
    使用 Sobel 算子计算多通道图像的梯度图。
    输入: image [b, c, h, w] - 输入图像 (batch_size, channels, height, width)
    输出: grad_mag [b, 1, h, w] - 梯度幅度图 (单通道)
    """
    b, c, h, w = image.shape
    
    # 定义 Sobel 算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
    
    # 扩展 Sobel 算子以匹配图像的通道数
    sobel_x = sobel_x.repeat(c, 1, 1, 1)  # 形状变为 [c, 1, 3, 3]
    sobel_y = sobel_y.repeat(c, 1, 1, 1)  # 形状变为 [c, 1, 3, 3]
    
    # 对每个通道独立应用 Sobel 算子
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=c)  # [b, c, h, w]
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=c)  # [b, c, h, w]
    grad_x = torch.clamp(grad_x, -1e4, 1e4)  # 限制梯度范围
    grad_y = torch.clamp(grad_y, -1e4, 1e4)
    # 计算梯度幅度并聚合通道
    # print(grad_x)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # [b, c, h, w]
    grad_mag = grad_mag.sum(dim=1, keepdim=True)      # [b, 1, h, w]
    
    return grad_mag

# 基于梯度图的损失函数
def Grad2Loss(registered, target, attention_map):
    """
    基于梯度图计算加权损失，关注结构和大致分布。
    输入:
        registered [b, c, h, w] - 配准后的图像
        target [b, c, h, w] - 目标图像
        attention_map [b, c, h, w] - 注意力图，值范围 [0, 1]，形变越大值越大
    输出:
        loss - 标量损失值
    """
    # 计算配准图像和目标图像的梯度图
    grad_registered = compute_gradient(registered)
    grad_target = compute_gradient(target)
    
    # 计算加权 MSE 损失
    loss = torch.mean(attention_map * (grad_registered - grad_target) ** 2)
    return loss
class GradientSSIMLoss(nn.Module):
    def __init__(self, lambda_reg=0.01, window_size=1,gene_num=10):
        super().__init__()
        self.lambda_reg = lambda_reg
        
        # 初始化Sobel滤波器（单通道输入）
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        # Sobel核初始化
        self._init_sobel_kernels()
        
        # SSIM参数
        window = self._create_window(window_size)
        self.register_buffer('window', window)
        self.window_size = window_size

    def _init_sobel_kernels(self):
        """Sobel核初始化与冻结"""
        sobel_x = torch.tensor([
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_x.weight.data = sobel_x
        self.sobel_y.weight.data = sobel_y
        
        for param in self.parameters():
            param.requires_grad = False

    def _to_gray(self, x):
        """通用灰度转换（兼容任意通道数）"""
        if x.shape[1] == 3:  # RGB图像
            x=x[:, 0] * 0.2989 + x[:, 1] * 0.5870 + x[:, 2] * 0.1140
            return x.unsqueeze(1).contiguous()
        
        else:               # 多通道科学图像
            return x.mean(dim=1, keepdim=True).clone()
  # 通道维度求平均

    def _compute_gradients(self, x):
        """梯度幅值计算（自动处理通道维度）"""
        # 转换为单通道
        x_gray = self._to_gray(x)  # (B,1,H,W)
        
        # 梯度计算
        # print('梯度计算1',x.shape)
        # print('梯度计算2',x_gray.shape)
        grad_x = self.sobel_x(x_gray)
        grad_y = self.sobel_y(x_gray)
        
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

    def _ssim(self, img1, img2):
        """单通道SSIM计算"""
        # 确保输入为单通道
        if img1.size(1) != 1 or img2.size(1) != 1:
            raise ValueError("SSIM输入必须是单通道")
        img1 = img1.contiguous()  # 确保内存连续
        img2 = img2.contiguous()
        # 高斯滤波
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2)
        
        # 方差协方差计算
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2) - mu1_mu2
        
        # SSIM计算
        # 使用分离的常数定义
        C1 = torch.tensor(0.01**2, device=img1.device)
        C2 = torch.tensor(0.03**2, device=img1.device)
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()

    def _create_window(self, window_size):
        """高斯窗创建"""
        sigma = 1.5
        x = torch.arange(window_size, dtype=torch.float) - window_size//2
        gauss = torch.exp(-x.pow(2)/(2*sigma**2))
        gauss /= gauss.sum()
        
        window = gauss.ger(gauss).view(1, 1, window_size, window_size)
        return window

    def forward(self, warped, target, flow):
        """
        参数：
        warped : 配准图像 (B, C, H, W) C可为任意通道数
        target : 目标图像 (B, 3, H, W) 固定RGB三通道
        flow   : 形变场 (B, 2, H, W)
        """
        # 梯度特征提取
        warped_grad = self._compute_gradients(warped)  # (B,1,H,W)
        target_grad = self._compute_gradients(target)  # (B,1,H,W)
        
        # 损失计算
        ssim_loss = self._ssim(warped_grad, target_grad)
        reg_loss = self._flow_regularization(flow)
        
        return ssim_loss + self.lambda_reg * reg_loss

    def _flow_regularization(self, flow):
        """形变场正则化"""
        # 空间梯度计算
        dx = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])  # y方向
        dy = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])  # x方向
        return torch.mean(dx**2) + torch.mean(dy**2)

import copy
import torch.nn.functional as F
def cross_entropy(logits,labels):
    """

    Args:
        logits: [2*BS, Cls_num] where Cls_num=2*BS
        labels: [2*BS, Cls_num] where Cls_num=2*BS
    Returns:

    """


    # Example inputs
    batch_size = logits.shape[0]

    # logits = torch.tensor([[1.2, 0.5, 2.1, 0.8, 1.7],
    #                        [1.1, 1.0, 1.3, 2.1, 0.9],
    #                        [2.2, 1.1, 0.9, 1.8, 0.7]])
    # labels = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])

    # Step 1: Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)

    # Step 2: Take the log of the probabilities
    log_probs = torch.log(probabilities)

    # Step 3: Gather the log-probabilities for the correct class labels
    # Use labels as index to pick the log probabilities for the correct classes

    correct_log_probs = []
    for i in range(batch_size):
        ce_persample = torch.sum(log_probs[i] * labels[i])
        correct_log_probs.append(ce_persample)
    correct_log_probs = torch.tensor(correct_log_probs, device=logits.device,requires_grad=True)
    # Step 4: Compute the negative log-likelihood
    negative_log_likelihood = -correct_log_probs

    # Step 5: Calculate the mean loss across the batch
    loss = negative_log_likelihood.mean()
    return loss

class SpatialTransformer(nn.Module):
    """ 空间变换器（保持与之前相同） """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        
        # 创建标准化网格
        vectors = [torch.linspace(-1, 1, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # shape (dim, H, W, D)
        grid = grid.unsqueeze(0)   # shape (1, dim, H, W, D)
        self.register_buffer('grid', grid)
        
    def forward(self, src, flow):
        
        # 新的网格点 = 原始网格 + 位移场
        new_locs = self.grid + flow
        # print(new_locs.shape)
        # 调整维度顺序以适应PyTorch的样本格式
        if len(src.shape) == 4:  # 2D
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]  # 调整坐标顺序为(y, x)
        elif len(src.shape) == 5:  # 3D
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]  # 调整顺序为(z, y, x)
        
        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)

class VecInt(nn.Module):
    """ 速度场积分模块 """
    def __init__(self, inshape, nsteps=7):
        super().__init__()
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, flow):
        for _ in range(self.nsteps):
            
            flow = flow + self.transformer(flow, flow) * self.scale
        return flow

class RescaleTransform(nn.Module):
    """ 形变场缩放模块 """
    def __init__(self, factor, mode='bilinear'):
        super().__init__()
        self.factor = factor
        self.mode = mode
        
    def forward(self, x):
        if self.factor < 1:
            return F.interpolate(x, scale_factor=self.factor,mode=self.mode, align_corners=True)
        else:
            return F.interpolate(x, scale_factor=self.factor,mode=self.mode, align_corners=True)

class VxmDense(nn.Module):
    """ 单向配准核心网络 """
    def __init__(self, 
                 inshape,
                 nb_unet_features=[[16,32,32,16], [16,32,32,16]],  # 编码器/解码器特征
                 int_steps=7,
                 svf_resolution=1,
                 int_resolution=2,
                 gene_num=10,
                 use_probs=False):
        super().__init__()
        
        # # 维度验证
        ndims = len(inshape)
        assert ndims in [2,3], "仅支持2D/3D数据"
        
        # 核心UNet结构
        self.unet = UNet_grid (
            in_channels=gene_num+3,  # 源+目标图像拼接
            # out_channels=ndims,
            # features=nb_unet_features,
            # conv_layers_per_stage=2
        )
        
        # 概率形变场
        self.use_probs = use_probs
        if use_probs:
            self.flow_logsigma = nn.Conv2d if ndims==2 else nn.Conv3d
            self.flow_logsigma = self.flow_logsigma(self.unet.out_channels, ndims, 3, padding=1)
            nn.init.normal_(self.flow_logsigma.weight, mean=0, std=1e-10)
            nn.init.constant_(self.flow_logsigma.bias, -10)
        
        # 分辨率调整
        self.svf_rescale = RescaleTransform(1/svf_resolution) if svf_resolution!=1 else None
        self.int_rescale = RescaleTransform(1/int_resolution) if int_resolution!=1 else None
        
        # 积分模块
        self.integrator = VecInt(
            inshape=[int(dim/int_resolution) for dim in inshape],
            nsteps=int_steps
        ) if int_steps>0 else None
        
        # 最终空间变换器
        self.transformer = SpatialTransformer(inshape)
    
    def forward(self, source, target):
        # 输入拼接

        x = torch.cat([source, target], dim=1)
        
        # UNet预测初始流场
        flow = self.unet(x)
        
        # 概率采样
        if self.use_probs:
            log_sigma = self.flow_logsigma(x)
            flow = flow + torch.exp(log_sigma) * torch.randn_like(log_sigma)
        
        # 调整SVF分辨率
        if self.svf_rescale:
            flow = self.svf_rescale(flow)
        
        # 保留SVF用于正则化
        svf = flow
        # print(flow.shape)
        # 调整积分分辨率
        if self.int_rescale:
            flow = self.int_rescale(flow)
        # flow = F.interpolate(flow, scale_factor=0.5, mode='bilinear')

        # 速度场积分
        if self.integrator:
            flow = self.integrator(flow)
        
        # 上采样到原始分辨率
        if self.int_rescale:
            flow = F.interpolate(flow, size=source.shape[2:], mode='bilinear')
        
        # 空间变换
        warped = self.transformer(source, flow)
        
        return warped, flow  # 返回配准后图像和形变场
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 核心卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False, kernel_init='he_normal'):
        super().__init__()
        self.residual = residual
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        # 初始化权重
        if kernel_init == 'he_normal':
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
            if self.shortcut is not None:
                nn.init.kaiming_normal_(self.shortcut.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.residual:
            if self.shortcut is not None:
                identity = self.shortcut(identity)
            out += identity
        
        return F.relu(out)

# 上采样块（包含跳跃连接）
class UpsampleBlock(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
def CC_ContrastiveLoss(WSI_feture, LRST_feature,temperature=0.07):
    """
    WSI_feture: Embeddings from the WSI encoder (batch_size, embedding_dim)
    LRST_feature: Embeddings from the LRST encoder (batch_size, embedding_dim)
    """
    # Normalize the embeddings to unit vectors

    WSI_feture = F.normalize(WSI_feture, p=2, dim=-1)
    LRST_feature = F.normalize(LRST_feature, p=2, dim=-1)

    WSI_feture_new=torch.concat((WSI_feture,LRST_feature),dim=0)
    LRST_feture_new = torch.concat((LRST_feature, WSI_feture), dim=0)
    # Compute the similarity matrix (dot product between image and text embeddings)
    logits_new = torch.matmul(WSI_feture_new, LRST_feture_new.T) / temperature

    # Create labels (positive pairs are diagonal elements)
    batch_size = WSI_feture_new.shape[0]
    labels = torch.zeros(size=(batch_size,batch_size), device=WSI_feture.device)
    for i in range(batch_size):
        labels[i,i]=1
    for i in range(int(batch_size/2)):
        labels[i, i+int(batch_size/2)] = 1
    for i in range(int(batch_size/2)):
        labels[i+int(batch_size/2),i] = 1
    labels.requires_grad=True
    # Calculate cross-entropy loss
    loss = cross_entropy(logits_new, labels)


    return loss




def CM_ContrastiveLoss(WSI_feture, LRST_feature,temperature=0.07):
    """
    WSI_feture: Embeddings from the WSI encoder (batch_size, embedding_dim)
    LRST_feature: Embeddings from the LRST encoder (batch_size, embedding_dim)
    """
    # Normalize the embeddings to unit vectors

    WSI_feture = F.normalize(WSI_feture, p=2, dim=-1)
    LRST_feature = F.normalize(LRST_feature, p=2, dim=-1)

    WSI_feture_new=torch.concat((WSI_feture,LRST_feature),dim=0)
    LRST_feture_new = torch.concat((LRST_feature, WSI_feture), dim=0)
    # Compute the similarity matrix (dot product between image and text embeddings)
    logits_new = torch.matmul(WSI_feture_new, LRST_feture_new.T) / temperature

    # Create labels (positive pairs are diagonal elements)
    batch_size = WSI_feture_new.shape[0]
    labels = torch.zeros(size=(batch_size,batch_size), device=WSI_feture.device)

    for i in range(int(batch_size/2)):
        for j in range(int(batch_size / 2),batch_size):
            labels[i,j]=1
    for i in range(int(batch_size / 2),batch_size):
        for j in range(batch_size):
            labels[i,j]=1
    labels.requires_grad = True
    # Calculate cross-entropy loss
    loss = cross_entropy(logits_new, labels)


    return loss

def IS_ContrastiveLoss(WSI_C, WSI_M,temperature=0.07):
    """
    WSI_feture: Embeddings from the WSI encoder (batch_size, embedding_dim)
    LRST_feature: Embeddings from the LRST encoder (batch_size, embedding_dim)
    """
    # Normalize the embeddings to unit vectors

    WSI_C = F.normalize(WSI_C, p=2, dim=-1)
    WSI_M = F.normalize(WSI_M, p=2, dim=-1)


    # Compute the similarity matrix (dot product between image and text embeddings)
    logits_new = torch.matmul(WSI_C, WSI_M.T) / temperature

    # Create labels (positive pairs are diagonal elements)
    batch_size = WSI_C.shape[0]
    labels = torch.zeros(size=(batch_size,batch_size), device=WSI_C.device)
    for i in range(batch_size):
        labels[i,i]=1

    labels.requires_grad=True
    # Calculate cross-entropy loss
    loss = cross_entropy(logits_new, labels)


    return loss

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
            self,
            spacial_dim: int,
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):

        
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class SE_Attention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):

        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for weighting specific regions in the image.
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)  # 7x7 conv
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute spatial attention weights
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)  # [batch, 2, H, W]
        attn = self.conv1(attn)  # [batch, 1, H, W]
        return self.sigmoid(attn) * x  # Apply attention weights to input


class GeneRelationWeighting(nn.Module):
    """
    Module to adjust ST gene features based on a gene relationship matrix.
    """
    def __init__(self, num_genes, gene_relation_matrix=None):
        super(GeneRelationWeighting, self).__init__()
        gene_relation_matrix = torch.tensor(gene_relation_matrix, dtype=torch.float32)

        self.gene_relation_matrix = nn.Parameter(gene_relation_matrix)

    def forward(self, gene_features):
        """
        Args:
            gene_features: [batch, num_genes, H, W] tensor
        Returns:
            Weighted gene features
        """
        # Reshape gene features for matrix multiplication
        batch, num_genes, H, W = gene_features.shape
        gene_features = gene_features.view(batch, num_genes, -1)  # [batch, num_genes, H*W]
        # Apply relationship matrix
        weighted_features = torch.matmul(self.gene_relation_matrix, gene_features)  # [batch, num_genes, H*W]
        return weighted_features.view(batch, num_genes, H, W)


class UNetModel(nn.Module):


    def __init__(
            self,
            gene_num,
            model_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,#是否使用残差块进行上采样和下采样。
            use_new_attention_order=False,#是否使用新型注意力模式
            root=''
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.gene_num = gene_num #基因数量，用于定义输入和输出通道数。
        # self.in_channels = in_channels
        self.model_channels = model_channels
        # self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions #指定在何种下采样率下使用注意力机制
        self.dropout = dropout
        self.channel_mult = channel_mult #通道数乘子，控制不同层的通道数变化
        self.conv_resample = conv_resample#是否使用卷积进行上采样和下采样。
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        

        time_embed_dim = model_channels * 4 #时间嵌入层用于处理时间步信息，生成一个时间嵌入向量，该向量随后会被注入到网络中的多个位置。
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        
        # co_expression = np.load(root + 'gene_coexpre.npy')
        self.pre = nn.Sequential(
            conv_nd(dims, model_channels, self.gene_num, 3, padding=1),
            nn.SiLU()
        )
        self.post = nn.Sequential(
            conv_nd(dims, self.gene_num, model_channels, 3, padding=1),
            nn.SiLU()
        )
        # self.gc1 = GraphConvolution(26*26, 26*26,co_expression,self.gene_num)#使用图卷积网络处理基因共表达数据。共表达矩阵存储在 gene_coexpre.npy 文件中

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, self.gene_num, ch, 3, padding=1))]
        )

        self.input_blocks_WSI5120 = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, 3, ch, 3, padding=1))]
        )

        self.input_blocks_WSI320 = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, 3, ch, 3, padding=1))]
        )

        hint_channels = model_channels
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch,time_embed_dim,dropout,out_channels=int(mult * model_channels),dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,)]
                ch = int(mult * model_channels)
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_blocks_WSI5120.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch,time_embed_dim,dropout,out_channels=out_ch,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,down=True,)
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                self.input_blocks_WSI5120.append(
                    TimestepEmbedSequential(ResBlock(ch,time_embed_dim,dropout,out_channels=out_ch,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,down=True,)
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(hint_channels,time_embed_dim,dropout,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich,time_embed_dim,dropout,out_channels=int(model_channels * mult),dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,)]
                ch = int(model_channels * mult)
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock(ch,time_embed_dim,dropout,out_channels=out_ch,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,up=True,)
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        conv_ch = self.channel_mult[-1] * self.model_channels

        self.input_blocks_lr = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        # self.input_blocks_WSI320 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks_WSI5120])


        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, self.gene_num*2, 3, padding=1)),
        )


        self.change_layer = nn.Sequential(
            conv_nd(dims, 128+2*self.gene_num, hint_channels, 3, padding=1),
            nn.SiLU()
        )


    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps,low_res, WSI_5120, Gene_index_map):#new1.5:, gene_class, Gene_index_map#WSI_320, gene_class,
        """
        Apply the model to an input batch.
        :param x: an [N x 20 x 256 x 256] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param low_res: [N x 20 x 26 x 26] round -13
        :param WSI_5120: [N x 3 x 256 256] 0-255
        :return: an [N x 40 x 256 x 256] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        WSI_5120=WSI_5120/255

        h_x = x.type(self.dtype)  ## backward noise of SR ST   
        h_spot = low_res.type(self.dtype)## spot ST            
        h_5120WSI = WSI_5120.type(self.dtype)## #              
        h_spot = F.interpolate(h_spot,(h_x.shape[2],h_x.shape[3]))
        for idx in range(len(self.input_blocks)):
            h_x = self.input_blocks[idx](h_x, emb) # 
            h_spot = self.input_blocks_lr[idx](h_spot, emb) # 
            h_5120WSI = self.input_blocks_WSI5120[idx](h_5120WSI, emb) # 
            hs.append((1 / 3) * h_x + (1 / 3) * F.interpolate(h_spot,(h_x.shape[2],h_x.shape[3])) + (1 / 3) * h_5120WSI)
        Gene_index_map = Gene_index_map.float()
        Final_merge=torch.concat((h_5120WSI,h_spot,Gene_index_map,Gene_index_map),dim=1)

        h = self.change_layer(Final_merge)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h2 = hs.pop()
            h = th.cat([h, h2], dim=1)
            h = module(h, emb)

        return self.out(h)


