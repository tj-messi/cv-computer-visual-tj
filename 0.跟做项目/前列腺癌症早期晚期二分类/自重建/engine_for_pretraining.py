# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import torch
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn
import utils
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from einops import rearrange
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def denormalize(images, mean, std):
    """将归一化图像反归一化到 [0, 1]"""
    mean = torch.tensor(mean).to(images.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(std).to(images.device).view(1, -1, 1, 1, 1)
    return images * std + mean

def visualize_mae_reconstruction(images, bool_masked_pos, mask_outputs, mask_labels, patch_size=16, tubelet_size=2, save_path=None, show_frames=False):
    """
    可视化 MAE 的重建效果，针对 90% 掩码比例
    Args:
        images: 输入视频，形状 [B, C, T, H, W]
        bool_masked_pos: 掩码位置，形状 [B, N]
        mask_outputs: 模型重建输出，形状 [B, N_m, C]
        mask_labels: 真实标签（掩码 patch），形状 [B, N_m, C]
        patch_size: 空间 patch 大小
        tubelet_size: 时间 tubelet 大小
        save_path: 保存路径（可选）
        show_frames: 是否显示每样本的 8 帧（默认 False，平均时间维度）
    """
    # 移动到 CPU 并转换为 float32
    images = images.cpu().to(torch.float32)
    bool_masked_pos = bool_masked_pos.cpu()
    mask_outputs = mask_outputs.cpu().to(torch.float32)  # 确保 float32
    mask_labels = mask_labels.cpu().to(torch.float32)  # 确保 float32

    # 打印形状和 dtype 以调试
    # print(f"visualize_mae_reconstruction: images shape = {images.shape}, dtype = {images.dtype}")
    # print(f"visualize_mae_reconstruction: bool_masked_pos shape = {bool_masked_pos.shape}, dtype = {bool_masked_pos.dtype}")
    # print(f"visualize_mae_reconstruction: mask_outputs shape = {mask_outputs.shape}, dtype = {mask_outputs.dtype}")
    # print(f"visualize_mae_reconstruction: mask_labels shape = {mask_labels.shape}, dtype = {mask_labels.dtype}")

    # 反归一化原始图像
    IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
    images_denorm = denormalize(images, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD).clamp(0, 1).detach()  # 分离梯度

    # 获取图像维度
    B, C, T, H, W = images.shape
    num_patches = (T // tubelet_size) * (H // patch_size) * (W // patch_size)  # 总 patch 数，例如 1568
    patch_dim = C * tubelet_size * patch_size * patch_size  # 每个 patch 的特征维度，例如 1536

    # 验证形状
    assert mask_outputs.shape == mask_labels.shape, f"mask_outputs shape {mask_outputs.shape} != mask_labels shape {mask_labels.shape}"
    assert mask_outputs.shape[1] <= num_patches, f"mask_outputs has {mask_outputs.shape[1]} patches, expected <= {num_patches}"
    assert mask_outputs.shape[2] == patch_dim, f"mask_outputs dim {mask_outputs.shape[2]} != patch_dim {patch_dim}"
    assert bool_masked_pos.shape == (B, num_patches), f"bool_masked_pos shape {bool_masked_pos.shape} != expected {(B, num_patches)}"

    # 初始化重建图像的 patch
    reconstructed_patches = torch.zeros(B, num_patches, patch_dim, dtype=torch.float32)  # 显式指定 float32

    # 获取掩码索引
    masked_indices = bool_masked_pos.bool()  # [B, N]，例如 [4, 1568]

    # 验证掩码 patch 数
    expected_masked_patches = mask_outputs.shape[1] * B  # 例如 1412 * 4 = 5648
    if masked_indices.sum() != expected_masked_patches:
        print(f"Warning: masked_indices.sum()={masked_indices.sum()}, expected {expected_masked_patches}")
        # 截断 masked_indices 以匹配 mask_outputs
        masked_indices = masked_indices[:, :mask_outputs.shape[1]]

    # 填充重建 patch
    mask_outputs_flat = mask_outputs.view(-1, patch_dim).to(torch.float32)  # [B * N_m, patch_dim]，例如 [5648, 1536]，确保 float32
    reconstructed_patches.view(B * num_patches, patch_dim)[masked_indices.view(-1)] = mask_outputs_flat

    # 将 patch 重新排列为图像
    reconstructed_images = rearrange(
        reconstructed_patches,
        'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)',
        t=T // tubelet_size,
        h=H // patch_size,
        w=W // patch_size,
        p0=tubelet_size,
        p1=patch_size,
        p2=patch_size,
        c=C
    ).clamp(0, 1).detach()  # 分离梯度

    # 创建掩码图像（掩码区域显示为灰色）
    mask_image = images_denorm.clone().detach()  # 确保分离梯度
    mask_patches = rearrange(
        mask_image,
        'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
        p0=tubelet_size,
        p1=patch_size,
        p2=patch_size
    )
    mask_patches[masked_indices] = 0.0  # 灰色
    mask_image = rearrange(
        mask_patches,
        'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)',
        t=T // tubelet_size,
        h=H // patch_size,
        w=W // patch_size,
        p0=tubelet_size,
        p1=patch_size,
        p2=patch_size,
        c=C
    ).detach()  # 分离梯度

    if show_frames:
        # 显示每样本的 8 帧（每 2 帧取 1 帧）
        fig, axes = plt.subplots(B, 4 * 8, figsize=(16 * 8, 4 * B))  # 每样本 8 帧，4 种图像
        if B == 1:
            axes = [axes]
        for i in range(B):
            for j in range(8):  # 显示 8 帧
                frame_idx = j * 2  # 每 2 帧取 1 帧（tubelet_size=2）
                # 原始图像
                axes[i][j*4+0].imshow(images_denorm[i][:, frame_idx].permute(1, 2, 0).detach().numpy())
                axes[i][j*4+0].set_title(f"Frame {frame_idx}")
                axes[i][j*4+0].axis('off')
                # 掩码图像
                axes[i][j*4+1].imshow(mask_image[i][:, frame_idx].permute(1, 2, 0).detach().numpy())
                axes[i][j*4+1].set_title(f"Masked {frame_idx}")
                axes[i][j*4+1].axis('off')
                # 重建图像
                axes[i][j*4+2].imshow(reconstructed_images[i][:, frame_idx].permute(1, 2, 0).detach().numpy())
                axes[i][j*4+2].set_title(f"Recon {frame_idx}")
                axes[i][j*4+2].axis('off')
                # 差异图像
                diff = torch.abs(reconstructed_images[i][:, frame_idx] - images_denorm[i][:, frame_idx]).detach()
                axes[i][j*4+3].imshow(diff.permute(1, 2, 0).numpy())
                axes[i][j*4+3].set_title(f"Diff {frame_idx}")
                axes[i][j*4+3].axis('off')
    else:
        # 默认：平均时间维度，显示单张 2D 图像
        fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))
        if B == 1:
            axes = [axes]
        for i in range(B):
            # 原始图像（取时间维度的平均）
            axes[i][0].imshow(images_denorm[i].permute(1, 2, 3, 0).mean(dim=0).detach().numpy())
            axes[i][0].set_title("Original")
            axes[i][0].axis('off')
            # 掩码图像
            axes[i][1].imshow(mask_image[i].permute(1, 2, 3, 0).mean(dim=0).detach().numpy())
            axes[i][1].set_title("Masked (90%)")
            axes[i][1].axis('off')
            # 重建图像
            axes[i][2].imshow(reconstructed_images[i].permute(1, 2, 3, 0).mean(dim=0).detach().numpy())
            axes[i][2].set_title("Reconstructed")
            axes[i][2].axis('off')
            # 差异图像
            diff = torch.abs(reconstructed_images[i] - images_denorm[i]).detach()
            axes[i][3].imshow(diff.permute(1, 2, 3, 0).mean(dim=0).numpy())
            axes[i][3].set_title("Difference")
            axes[i][3].axis('off')

    plt.tight_layout()

    # 确保保存路径存在
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)  # 释放内存

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_ssim_psnr(original, reconstructed):
    """
    Compute SSIM and PSNR between two video sequences.
    The input shape is (B, C, T, H, W) where:
        - B: Batch size
        - C: Number of channels (3 for RGB)
        - T: Number of frames (16 in this case)
        - H: Height of each frame
        - W: Width of each frame
    
    Args:
    - original (torch.Tensor): Original video tensor, shape (B, C, T, H, W), range [0, 1]
    - reconstructed (torch.Tensor): Reconstructed video tensor, shape (B, C, T, H, W), range [0, 1]
    
    Returns:
    - ssim_value (float): Mean SSIM score across all frames in the batch
    - psnr_value (float): Mean PSNR score across all frames in the batch
    """
    # Convert to numpy arrays and rearrange to (B*T, C, H, W) format for processing
    original = original.detach().cpu().numpy().transpose(0, 2, 3, 4, 1).reshape(-1, original.shape[1], original.shape[3], original.shape[4])  # (B*T, C, H, W)
    reconstructed = reconstructed.detach().cpu().numpy().transpose(0, 2, 3, 4, 1).reshape(-1, reconstructed.shape[1], reconstructed.shape[3], reconstructed.shape[4])  # (B*T, C, H, W)
    
    # Ensure the images are in the correct range for PSNR and SSIM (0-255)
    original = np.clip(original * 255, 0, 255).astype(np.uint8)
    reconstructed = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)

    # Calculate SSIM and PSNR for each frame in the batch
    batch_ssim = []
    batch_psnr = []
    for i in range(original.shape[0]):
        # Specify window size and channel_axis to fix the error
        ssim_value = ssim(original[i], reconstructed[i], multichannel=True, win_size=3, channel_axis=-1)
        psnr_value = psnr(original[i], reconstructed[i])
        
        batch_ssim.append(ssim_value)
        batch_psnr.append(psnr_value)
    
    # Return the average SSIM and PSNR across all frames in the batch
    return np.mean(batch_ssim), np.mean(batch_psnr)



def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    patch_size: int = 16,
                    normlize_target: bool = True,
                    log_writer=None,
                    lr_scheduler=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    for step, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group[
                        "lr_scale"]
                if wd_schedule_values is not None and param_group[
                        "weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        
        # NOTE: When the decoder mask ratio is 0,
        # in other words, when decoder masking is not used,
        # decode_masked_pos = ~bool_masked_pos
        images, bool_masked_pos, decode_masked_pos = batch

        images = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).view(2, -1, 3).to(torch.int16)
        
        '''
        # visualize the data
        max_data = 0 
        min_data = 4
        for data in bool_masked_pos:
            flattened_data = data.flatten()
    
            # 更新最大值和最小值
            max_data = max(max_data, flattened_data.max().item())  # 使用 `.item()` 获取标量
            min_data = min(min_data, flattened_data.min().item())  # 使用 `.item()` 获取标量
        print("max :" ,max_data , "min : " , min_data)
        max : 3 min :  0
        '''
        decode_masked_pos = decode_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :,
                                                                     None,
                                                                     None,
                                                                     None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :,
                                                                   None, None,
                                                                   None]
            unnorm_images = images * std + mean  # in [0, 1]

            # print("unnorm_images:" , unnorm_images.max())

            if not normlize_target:
                images_squeeze = rearrange(
                    unnorm_images,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                    p0=2,
                    p1=patch_size,
                    p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(
                    dim=-2, keepdim=True)) / (
                        images_squeeze.var(
                            dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')

            else:
                images_patch = rearrange(
                    unnorm_images,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                    p0=2,
                    p1=patch_size,
                    p2=patch_size)

            # print("images_patch",images_patch.max())
            B, N, C = images_patch.shape
            labels = images_patch[~decode_masked_pos].reshape(B, -1, C)

        tubelet_size = 2
        if loss_scaler is None:
            outputs = model(images, bool_masked_pos, decode_masked_pos)
            loss = (outputs - labels)**2
            loss = loss.mean(dim=-1)
            cal_loss_mask = bool_masked_pos[~decode_masked_pos].reshape(B, -1)
            loss = (loss * cal_loss_mask).sum() / cal_loss_mask.sum()

            if step % 100 == 0:  # 每100步可视化一次
                with torch.cuda.amp.autocast():
                    visualize_mae_reconstruction(
                        images, bool_masked_pos, outputs, labels,
                        patch_size=patch_size, save_path=f'results/epoch_{epoch}/recon_step_{step}.png'
                    )

        else:
            with torch.cuda.amp.autocast():
                # 动态获取当前 batch size
                B = images.shape[0]
                num_patches = (images.shape[2] // tubelet_size) * (images.shape[3] // patch_size) * (images.shape[4] // patch_size)

                # 确保 bool_masked_pos 的第一个维度与 B 一致
                if bool_masked_pos.shape[0] != B:
                    # print(f"Warning: bool_masked_pos batch size {bool_masked_pos.shape[0]} does not match images batch size {B}, adjusting...")
                    bool_masked_pos = bool_masked_pos[:B]

                # print("images:",images.max())

                mask_outputs, p_x, bool_masked_pos = model(images, bool_masked_pos, decode_masked_pos)

                # 打印形状以调试
                # print(f"Step {step}: images shape = {images.shape}")
                # print(f"Step {step}: bool_masked_pos shape = {bool_masked_pos.shape}")
                # print(f"Step {step}: mask_outputs shape = {mask_outputs.shape}")
                # print(f"Step {step}: mask_outputs dtype = {mask_outputs.dtype}")

                # mask_labels (lbls in the original order)
                mask_labels = images_patch[bool_masked_pos].reshape(B, -1, C)

                mask_labels = mask_labels.to(torch.float32)  # 转换为 float32
                mask_outputs = mask_outputs.to(torch.float32)  # 转换为 float32

                mask_labels_d = mask_labels.detach().cpu().numpy()
                mask_outputs_d = mask_outputs.detach().cpu().numpy()

                # print(mask_labels_d.dtype)
                # print(mask_outputs_d.dtype)

                # print(mask_labels_d.min(), mask_labels_d.max())
                # print(mask_outputs_d.min(), mask_outputs_d.max()) 
                
                mask_ssim = psnr(mask_labels_d,mask_outputs_d)
                # print(mask_ssim)

                # Reconstruction Loss
                loss_func = nn.MSELoss(reduction="none")
                mask_l_r = torch.mean(loss_func(input=mask_outputs, target=mask_labels), dim=-1)

                # Smoothness Loss 
                patch_dim = images.shape[1] * tubelet_size * patch_size * patch_size
                reconstructed_patches = torch.zeros(B, num_patches, patch_dim, dtype=torch.float32, device=images.device)

                masked_indices = bool_masked_pos.bool()
                # print(f"Step {step}: masked_indices shape = {masked_indices.shape}, total elements = {masked_indices.view(-1).shape}")

                mask_outputs_flat = mask_outputs.view(-1, patch_dim).to(torch.float32)  # 确保 float32
                # print(f"Step {step}: mask_outputs_flat shape = {mask_outputs_flat.shape}, dtype = {mask_outputs_flat.dtype}")
                # print(f"Step {step}: reconstructed_patches.view shape = {reconstructed_patches.view(B * num_patches, patch_dim).shape}")

                # 填充重建 patch
                expected_elements = mask_outputs_flat.shape[0]  # B * 1412
                if masked_indices.sum() != expected_elements:
                    print(f"Warning: masked_indices.sum()={masked_indices.sum()}, expected {expected_elements}")
                    masked_indices_flat = masked_indices.view(-1)[:expected_elements]
                    reconstructed_patches.view(B * num_patches, patch_dim)[:expected_elements][masked_indices_flat] = mask_outputs_flat
                else:
                    reconstructed_patches.view(B * num_patches, patch_dim)[masked_indices.view(-1)] = mask_outputs_flat

                # 重排列为视频格式
                reconstructed_images = rearrange(
                    reconstructed_patches,
                    'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)',
                    t=images.shape[2] // tubelet_size,
                    h=images.shape[3] // patch_size,
                    w=images.shape[4] // patch_size,
                    p0=tubelet_size,
                    p1=patch_size,
                    p2=patch_size,
                    c=images.shape[1]
                ).clamp(0, 1)

                # 计算平滑损失
                diff_h = (reconstructed_images[:, :, :, 1:, :] - reconstructed_images[:, :, :, :-1, :]).pow(2).mean()
                diff_w = (reconstructed_images[:, :, :, :, 1:] - reconstructed_images[:, :, :, :, :-1]).pow(2).mean()
                smoothness_loss = diff_h + diff_w 
                # m_l_smooth = 2.0 * smoothness_loss  # 平滑损失权重，可调整

                # Visualization
                if step % 100 == 0:
                    visualize_mae_reconstruction(
                        images, bool_masked_pos, mask_outputs, mask_labels,
                        patch_size=patch_size,
                        tubelet_size=2,
                        save_path=f'/media/tongji/VideoMAEv2-master/output/7-9-pt-alldata/results/epoch_{epoch}/recon_step_{step}.png',
                        show_frames=False
                    )

                    # 计算 SSIM 和 PSNR
                    ssim_value, psnr_value = compute_ssim_psnr(images, reconstructed_images)
                    print(f"Step {step}: SSIM = {ssim_value:.4f}, PSNR = {psnr_value:.4f}")

                # Sampling Loss
                l_s = torch.zeros(images.shape[0], ).to(mask_l_r.device)
                for i in range(p_x.shape[0]):
                    m = torch.distributions.categorical.Categorical(probs=p_x[i])
                    log_probs = m.log_prob(torch.arange(0, p_x.shape[1], 1).to(p_x.device))
                    mask_log_probs = log_probs[bool_masked_pos[i]]
                    l_s[i] = -torch.mean(mask_log_probs * mask_l_r[i].detach())

                # Total Loss
                m_l_r = torch.mean(mask_l_r)
                m_l_s = 1e-4 * torch.mean(l_s)
                loss = m_l_r + m_l_s 

                # print(f"Step {step}: Reconstruction Loss = {m_l_r.item()}, Sampling Loss = {m_l_s.item()}, Smoothness Loss = {m_l_smooth.item()}")

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(2)

        optimizer.zero_grad()

        if loss_scaler is None:
            loss.backward()
            if max_norm is None:
                grad_norm = utils.get_grad_norm_(model.parameters())
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm)
            optimizer.step()
            loss_scale_value = 0
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]



        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

'''import torch
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    patch_size: int = 16,
                    normlize_target: bool = True,
                    log_writer=None,
                    lr_scheduler=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    output_dir="/media/tongji/VideoMAEv2-master/reconstruct_pic"):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    os.makedirs(output_dir, exist_ok=True)  # 创建保存图像的目录

    for step, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        os.makedirs(os.path.join(output_dir,str(epoch)), exist_ok=True)  # 创建保存图像的目录
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        images, bool_masked_pos, decode_masked_pos = batch

        images = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)
        decode_masked_pos = decode_masked_pos.to(
            device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :,
                                                                     None,
                                                                     None,
                                                                     None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :,
                                                                   None, None,
                                                                   None]
            unnorm_images = images * std + mean  # in [0, 1]

            if normlize_target:
                images_squeeze = rearrange(
                    unnorm_images,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                    p0=2,
                    p1=patch_size,
                    p2=patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(
                    dim=-2, keepdim=True)) / (
                        images_squeeze.var(
                            dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(
                    unnorm_images,
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                    p0=2,
                    p1=patch_size,
                    p2=patch_size)

            B, N, C = images_patch.shape
            print(B,N,C)
            print(images_patch.shape)
            labels = images_patch[~decode_masked_pos].reshape(B, -1, C)

        if loss_scaler is None:
            outputs = model(images, bool_masked_pos, decode_masked_pos)
            loss = (outputs - labels)**2
            loss = loss.mean(dim=-1)
            cal_loss_mask = bool_masked_pos[~decode_masked_pos].reshape(B, -1)
            loss = (loss * cal_loss_mask).sum() / cal_loss_mask.sum()
        else:
            with torch.cuda.amp.autocast():
                outputs = model(images, bool_masked_pos, decode_masked_pos)
                loss = (outputs - labels)**2
                loss = loss.mean(dim=-1)
                cal_loss_mask = bool_masked_pos[~decode_masked_pos].reshape(
                    B, -1)
                loss = (loss * cal_loss_mask).sum() / cal_loss_mask.sum()

        # Save the images for debugging
        if step % 1 == 0:  # Save every 10 steps or any other interval
            # Save the original and reconstructed images
            reconstructed_image = outputs[0].cpu().detach()  # Take the first sample in the batch
            original_image = images[0].cpu().detach()

            B = 64  # 批量大小
            N = 1568  # 每个图像块数量（例如 28x28 个小块）
            C = 1536  # 图像的通道数（例如 RGB）
            patch_size = 16  # 每个图像块的大小（16x16）
            image_size = (224, 224)  # 原始图像大小（例如 224x224）
            reconstructed_image = unflatten_image(images_patch, patch_size, image_size)

            print(reconstructed_image.shape)
            print(original_image.shape)
            if original_image.dim() == 4:
                original_image = original_image.squeeze(0)
                print(original_image.dim())
            if reconstructed_image.dim() == 4:
                reconstructed_image = reconstructed_image.squeeze(0)
                print(reconstructed_image.dim())

            # Convert the image to float32 before saving
            reconstructed_image = reconstructed_image.to(torch.float32)
            reconstructed_image = reconstructed_image.mul(255).clamp(0, 255).to(torch.uint8)
            original_image = original_image.to(torch.float32)
            original_image = original_image.mul(255).clamp(0, 255).to(torch.uint8)

            save_image(reconstructed_image, os.path.join(output_dir,str(epoch), f'reconstructed_epoch{epoch}_step{step}.png'))
            save_image(original_image, os.path.join(output_dir,str(epoch) ,f'original_epoch{epoch}_step{step}.png'))

        # Continue with the existing training process...

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(2)

        optimizer.zero_grad()

        if loss_scaler is None:
            loss.backward()
            if max_norm is None:
                grad_norm = utils.get_grad_norm_(model.parameters())
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm)
            optimizer.step()
            loss_scale_value = 0
        else:
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def unflatten_image(images_patch, patch_size, image_size):
    """
    将展平后的图像块恢复回原始图像形状。

    :param images_patch: 展平后的图像块，形状为 (B, N, C)
    :param patch_size: 每个图像块的尺寸 (height, width)
    :param image_size: 原始图像的尺寸 (height, width)
    :return: 恢复后的图像，形状为 (B, C, H, W)
    """
    B, N, C = images_patch.shape
    H, W = image_size

    # 计算每行和每列的块数
    blocks_per_row = W // patch_size
    blocks_per_col = H // patch_size

    # 检查 N 是否与块数匹配
    expected_N = blocks_per_row * blocks_per_col
    if N != expected_N:
        print(f"Warning: Expected N ({expected_N}) does not match the actual N ({N})")
        print("Adjusting the expected block counts based on N...")

        # Adjust the number of blocks per row and col based on actual N
        blocks_per_row = int(N ** 0.5)
        blocks_per_col = int(N // blocks_per_row)

    # 恢复为 (B, C, H, W) 形状
    images_unflattened = images_patch.reshape(B, blocks_per_col, blocks_per_row, patch_size, patch_size, C)

    # 重新排列块的位置，将其恢复到原始图像的形状
    images_unflattened = images_unflattened.permute(0, 5, 1, 3, 2, 4)  # 调整维度顺序

    # 将块连接起来，恢复为 (B, C, H, W)
    images_reconstructed = images_unflattened.reshape(B, C, H, W)

    return images_reconstructed'''