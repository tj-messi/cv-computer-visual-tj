from abc import abstractmethod

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch
import sys
sys.path.append('/home/cmet-standard/Hanyu/JYX/new/TMI12.8/')
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

import copy
import torch.nn.functional as F

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

# class Com_loss(nn.Module):
#     def __init__(self, c=170, device='cuda'):
#         super(Com_loss, self).__init__()
#         self.c = c
#         self.device = device
#         self.hx = torch.tensor([[1/3, 0, -1/3]]*3, dtype=torch.float).unsqueeze(0).unsqueeze(0)
#         self.hx = self.hx.expand(64, 1, 3, 3).to(self.device)  # 扩展到64个通道
#         self.hy = self.hx.transpose(2, 3).to(self.device)
#         # 均值滤波核
#         self.ave_filter = torch.tensor([[0.25, 0.25], [0.25, 0.25]], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)
#         self.ave_filter = self.ave_filter.expand(64, 1, 2, 2).to(self.device)
#     def forward(self, dis_img, ref_img):
#         if torch.max(dis_img) <= 1:
#             dis_img = dis_img * 255
#         if torch.max(ref_img) <= 1:
#             ref_img = ref_img * 255

#         dis_img = dis_img.float()
#         ref_img = ref_img.float()

#         ave_dis = F.conv2d(dis_img, self.ave_filter, stride=1, groups=64)
#         ave_ref = F.conv2d(ref_img, self.ave_filter, stride=1, groups=64)

#         down_step = 2  # 下采样间隔
#         ave_dis_down = ave_dis[:, :, 0::down_step, 0::down_step]
#         ave_ref_down = ave_ref[:, :, 0::down_step, 0::down_step]

#         mr_sq = F.conv2d(ave_ref_down, self.hx, groups=64)**2 + F.conv2d(ave_ref_down, self.hy, groups=64)**2
#         md_sq = F.conv2d(ave_dis_down, self.hx, groups=64)**2 + F.conv2d(ave_dis_down, self.hy, groups=64)**2
#         mr = torch.sqrt(mr_sq)
#         md = torch.sqrt(md_sq)

#         GMS = (2 * mr * md + self.c) / (mr_sq + md_sq + self.c)
#         GMSD = torch.std(GMS.view(-1))
#         return GMSD

class GMSD_loss(nn.Module):
    def __init__(self, c=170, device='cuda', noise_std=0.01,num_channels=128):
        """
        c: GMSD公式中的常数
        device: 运行设备
        noise_std: 添加到超球空间特征的噪声标准差
        """
        super(GMSD_loss, self).__init__()
        self.c = c
        self.device = device
        self.noise_std = noise_std
        self.num_channels = num_channels

        # Sobel算子用于梯度计算
        self.hx = torch.tensor([[1/3, 0, -1/3]]*3, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        self.hx = self.hx.expand(num_channels, 1, 3, 3).to(self.device)
        self.hy = self.hx.transpose(2, 3).to(self.device)

        # 均值滤波核
        self.ave_filter = torch.tensor([[0.25, 0.25], [0.25, 0.25]], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)
        self.ave_filter = self.ave_filter.expand(num_channels, 1, 2, 2).to(self.device)

    def forward(self, dis_img, ref_img):
        """
        计算输入图像的GMSD loss, 并在特征映射中引入超球空间的增强。
        dis_img: 失真图像 [B, C, H, W]
        ref_img: 参考图像 [B, C, H, W]
        """
        if torch.max(dis_img) <= 1:
            dis_img = dis_img * 255
        if torch.max(ref_img) <= 1:
            ref_img = ref_img * 255

        dis_img = dis_img.float()
        ref_img = ref_img.float()

        # 均值滤波
        ave_dis = F.conv2d(dis_img, self.ave_filter, stride=1, groups=self.num_channels)
        ave_ref = F.conv2d(ref_img, self.ave_filter, stride=1, groups=self.num_channels)

        # 下采样
        down_step = 2
        ave_dis_down = ave_dis[:, :, 0::down_step, 0::down_step]
        ave_ref_down = ave_ref[:, :, 0::down_step, 0::down_step]

        # 超球空间映射
        ave_dis_down = self._map_to_hypersphere(ave_dis_down)
        ave_ref_down = self._map_to_hypersphere(ave_ref_down)

        # 梯度计算
        mr_sq = F.conv2d(ave_ref_down, self.hx, groups=self.num_channels)**2 + F.conv2d(ave_ref_down, self.hy, groups=self.num_channels)**2
        md_sq = F.conv2d(ave_dis_down, self.hx, groups=self.num_channels)**2 + F.conv2d(ave_dis_down, self.hy, groups=self.num_channels)**2
        mr = torch.sqrt(mr_sq)
        md = torch.sqrt(md_sq)

        # GMSD计算
        GMS = (2 * mr * md + self.c) / (mr_sq + md_sq + self.c)
        GMSD = torch.std(GMS.view(-1))
        return GMSD

    def _map_to_hypersphere(self, x):
        """
        将输入特征映射到超球空间并添加噪声。
        x: 输入特征 [B, C, H, W]
        """
        # 超球空间
        x_normalized = x / x.norm(dim=1, keepdim=True)  # [B, C, H, W]

        # 小范围噪声
        noise = torch.randn_like(x_normalized) * self.noise_std
        x_noisy = x_normalized + noise

        # 再次归一化
        x_noisy_normalized = x_noisy / x_noisy.norm(dim=1, keepdim=True)
        return x_noisy_normalized
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class OptimizedGMSD(nn.Module):
    """
    Combined GMSD Module with Spatial Attention and Gene Relation Weighting.
    """
    def __init__(self, base_gmsd):
        super(OptimizedGMSD, self).__init__()
        self.base_gmsd = base_gmsd  # Base GMSD implementation
        self.spatial_attention = SpatialAttention()
        self.gene_relation_weighting = GeneRelationWeighting(num_genes, gene_relation_matrix)

    def forward(self, ref_img, dis_img, gene_features):
        """
        Args:
            ref_img: High-resolution reference image [batch, C, H, W]
            dis_img: Low-resolution distorted image [batch, num_genes, H, W]
            gene_features: ST gene feature map [batch, num_genes, H, W]
        """
        # Apply spatial attention to reference and distorted images
        ref_img = self.spatial_attention(ref_img)
        # Apply gene relation weighting to ST gene features
        dis_img = self.gene_relation_weighting(gene_features)
        # Compute GMSD on processed images
        return self.base_gmsd(ref_img, dis_img)

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
        
        co_expression = np.load(root + 'gene_coexpre.npy')
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
            ResBlock(ch,time_embed_dim,dropout,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,),
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
        self.dim_reduction_non_zeros = nn.Sequential(
            conv_nd(dims, 2 * conv_ch, conv_ch, 1, padding=0),
            nn.SiLU()
        )

        self.conv_common = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.conv_distinct = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.fc_modulation_1 = nn.Sequential(
            nn.Linear(1024, 1024),
        )
        self.fc_modulation_2 = nn.Sequential(
            nn.Linear(1024, 1024),
        )

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, self.gene_num*2, 3, padding=1)),
        )

        self.to_q = nn.Linear(model_channels, model_channels, bias=False)
        self.to_k = nn.Linear(model_channels, model_channels, bias=False)
        self.to_v = nn.Linear(model_channels, model_channels, bias=False)

        self.to_q_con = nn.Linear(model_channels, model_channels, bias=False)
        self.to_k_con = nn.Linear(int(model_channels*1.5), model_channels, bias=False)
        self.to_v_con = nn.Linear(int(model_channels*1.5), model_channels, bias=False)
        self.replacer = FeatureNoiseReplacer(replacement_prob=0.8)  # 80%的概率替换
        
        # co_expression = np.load('./gene_coexpre.npy')
        co_expression = np.load(root + 'gene_coexpre.npy')
        co_expression_new =co_expression[0:self.gene_num*2,0:self.gene_num*2]
        # print(co_expression_new.shape)
        self.MemoryNetwork = GeneMemoryNetwork(gene_dim=64, num_genes=self.gene_num,query_dim=self.gene_num,memory_bank = co_expression_new)
        #self.optimized_gmsd = OptimizedGMSD(GMSD_loss(), self.gene_num, co_expression_new)
        self.GMSD_loss = GMSD_loss(num_channels=self.model_channels)
        self.conv_layer1 = nn.Sequential(
            conv_nd(dims, ch+self.gene_num, ch, 3, padding=1),
            nn.SiLU()
        )

        #self.conv_map = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv_layer2 = nn.Sequential(
            conv_nd(dims, input_ch+self.gene_num, input_ch, 3, padding=1),
            nn.SiLU()
        )

    def compute_contrastive_loss(self, gene_data):
        """
        gene_data: [BS, num_genes, H, W]
        多基因数据，支持样本内和样本间的对比学习。

        Returns:
            contrastive_loss: 对比损失 (NT-Xent loss)
        """
        batch_size, num_genes, H, W = gene_data.shape
        positive_pairs = []
        negative_pairs = []

        # 样本内对比（正样本对）
        for i in range(batch_size):
            for g1 in range(num_genes):
                for g2 in range(g1 + 1, num_genes):  # 避免自比较
                    pos = F.cosine_similarity(gene_data[i, g1], gene_data[i, g2])  # [H, W]
                    positive_pairs.append(pos)

        # 样本间对比（负样本对）
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:  # 避免和自己比较
                    for g1 in range(num_genes):
                        for g2 in range(num_genes):
                            neg = F.cosine_similarity(gene_data[i, g1], gene_data[j, g2])  # [H, W]
                            negative_pairs.append(neg)

        # 转换为张量
        positive_pairs = torch.stack(positive_pairs)  # [num_pos_pairs, H, W]
        negative_pairs = torch.stack(negative_pairs)  # [num_neg_pairs, H, W]

        # 对比损失计算 (NT-Xent Loss)
        pos_loss = -torch.log(
            torch.exp(positive_pairs) / (
                torch.exp(positive_pairs) + torch.sum(torch.exp(negative_pairs), dim=0)
            )
        )

        contrastive_loss = torch.mean(pos_loss)  # 平均所有正样本对的损失

        return contrastive_loss
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

    def forward(self, x, timesteps,low_res, WSI_5120,WSI_320, gene_class, Gene_index_map):#new1.5:, gene_class, Gene_index_map
        """
        Apply the model to an input batch.
        :param x: an [N x 50 x 256 x 256] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param ratio: 0-1
        :param low_res: [N x 50 x 26 x 26] round -13
        :param WSI_5120: [N x 3 x 256 256] 0-255
        :param WSI_320: [N x 256 x 3 x 16 16] 0-255
        :return: an [N x 50 x 256 x 256] Tensor of outputs.
        """


        ratio = x[0, 0, 0, 0]
        x = x[:, int(x.shape[1] / 2):x.shape[1], ...]  # [N x 50 x 256 x 256]

        # ratio=1 #

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        WSI_5120=WSI_5120/255
        WSI_320=th.reshape(WSI_320,(-1,WSI_320.shape[2],WSI_320.shape[3],WSI_320.shape[4]))/255 #[N.256 x 3 x 16 x 16]

        h_x = x.type(self.dtype)  ## backward noise of SR ST   [N x 50 x 256 x 256]  round -4
        h_spot = low_res.type(self.dtype)## spot ST            [N x 50 x 26 x 26] round -13
        h_5120WSI = WSI_5120.type(self.dtype)## #              [N x 3 x 256 x 256]
        h_320WSI = WSI_320.type(self.dtype)##  #              [N.256.ratio x 3 x 16 x 16]
        

###############概率掩码遮蔽############       
        
        h_spot = self.replacer.replace_with_noise(h_spot, ratio, dtype=torch.float32)

###############特征提取器######################
        # print('h_x',h_x.shape)
        # print('h_spot',h_x.shape)
        # print('h_5120WSI',h_x.shape)
        for idx in range(len(self.input_blocks)):

            h_x = self.input_blocks[idx](h_x, emb) # [N x 16 x 64 x 64]
            h_spot = self.input_blocks_lr[idx](h_spot, emb) # [N x 16 x 6 x 6]
            h_5120WSI = self.input_blocks_WSI5120[idx](h_5120WSI, emb) # [N x 16 x 64 x 64]
            hs.append((1 / 3) * h_x + (1 / 3) * F.interpolate(h_spot,(h_x.shape[2],h_x.shape[3])) + (1 / 3) * h_5120WSI)
        for idx in range(len(self.input_blocks_WSI320)):
            h_320WSI = self.input_blocks_WSI320[idx](h_320WSI, emb)

###########MEMORY BANK&&contrastive##################

        h_ori=h_temp = h_x

        h_temp=self.pre(h_temp)

        h_temp = self.MemoryNetwork(h_temp)
        h_temp = F.relu(h_temp)

        # gene_loss = self.compute_contrastive_loss(h_x.clone())
        h_temp = self.post(h_temp)
        h_x=h_ori*0.99+h_temp*0.01
########################################################################
        #########
        ######### entropy based cross attention
        #########
        if ratio == 2.0:
            ratio = 1.0

        h_320WSI = th.reshape(h_320WSI, (h_x.shape[0], -1, h_320WSI.shape[1], h_320WSI.shape[2], h_320WSI.shape[3]))
        h_320WSI = h_320WSI[:, 0:int(h_320WSI.shape[1] * ratio), ...]  # [N x 256.ratio x 16 x 16 x 16]
        h_320WSI = th.mean(h_320WSI, dim=1) # [N x 16 x 16 x 16]
        h_320WSI = F.interpolate(h_320WSI, size=(h_5120WSI.shape[2], h_5120WSI.shape[3])) # [N x 16 x 64 x 64]
        h_320WSI=th.reshape(h_320WSI,(h_320WSI.shape[0],h_320WSI.shape[1],-1))
        h_320WSI=th.transpose(h_320WSI,1,2) # [N x 4096 x 16]

        h_5120WSI_pre=th.reshape(h_5120WSI,(h_5120WSI.shape[0],h_5120WSI.shape[1],-1))
        h_5120WSI_pre = th.transpose(h_5120WSI_pre,1,2)  # [N x 4096 x 16]

        q = self.to_q(h_5120WSI_pre) # [N x 4096 x 16]
        k = self.to_k(h_320WSI) # [N x 4096 x 16]
        v = self.to_v(h_320WSI) # [N x 4096 x 16]
        mid_atten=torch.matmul(q,th.transpose(k,1,2))

        scale = q.shape[2] ** -0.5
        mid_atten=mid_atten*scale
        sfmax = nn.Softmax(dim=-1)#
        mid_atten=sfmax(mid_atten)# [N x 4096 x 4096]
        WSI_atten = torch.matmul(mid_atten, v) # [N x 4096 x 16]
        WSI_atten=th.transpose(WSI_atten,1,2)# [N x 16 x 4096 ]
        WSI_atten = th.reshape(WSI_atten, (WSI_atten.shape[0],WSI_atten.shape[1], h_5120WSI.shape[2], h_5120WSI.shape[3]))# [N x 16 x 64 x64 ]

        ### weight
        WSI_atten=0.9*h_5120WSI+0.1*WSI_atten
        ######### Local Gradient Similarity Comparison

        GMSD_loss = self.GMSD_loss(WSI_atten.clone().cuda(), h_spot.clone().cuda())
        
        #########
        ######### Disentangle and modulation and cross atten
        #########
        com_WSI = self.conv_common(WSI_atten) # [N x 8 x 64 x 64]
        com_spot = self.conv_common(h_spot)
        com_spot =F.interpolate(com_spot, size=(WSI_atten.shape[2], WSI_atten.shape[3])) # [N x 8 x 64 x 64]

        dist_WSI = self.conv_distinct(WSI_atten) # [N x 8 x 64 x 64]
        dist_spot = self.conv_distinct(h_spot)
        dist_spot = F.interpolate(dist_spot, size=(WSI_atten.shape[2], WSI_atten.shape[3]))  # [N x 8 x 64 x 64]

        com_h = (1 / 2) * com_WSI + (1 / 2) * com_spot # [N x 8 x 64 x 64]

        ##modulatoion
        part=2
        part_width=int(dist_WSI.shape[2]/part)
        WSI_part_dist=dist_WSI
        spot_part_dist = dist_spot
        for i in range(part):
            for j in range(part):
                WSI_part=dist_WSI[...,i*part_width:(i+1)*part_width,j*part_width:(j+1)*part_width] # [N x 8 x 32 x 32]
                spot_part = dist_spot[..., i * part_width:(i + 1) * part_width, j * part_width:(j + 1) * part_width] # [N x 8 x 32 x 32]
                WSI_part = th.reshape(WSI_part, (WSI_part.shape[0], WSI_part.shape[1], -1)) # [N x 8 x 1024]
                spot_part = th.reshape(spot_part, (spot_part.shape[0], spot_part.shape[1], -1))  # [N x 8 x 1024]
                WSI_part_T = th.transpose(WSI_part, 1, 2)  # [N x 1024 x 8 ]
                spot_part_T = th.transpose(spot_part, 1, 2)  # [N x 1024 x 8 ]

                F_WSItoSpot=th.matmul(spot_part_T,WSI_part)# [N x 1024 x 1024]
                w_WSItoSpot=self.fc_modulation_1(F_WSItoSpot)# [N x 1024 x 1024]
                sfmax_module = nn.Softmax(dim=-1)
                w_WSItoSpot=sfmax_module(w_WSItoSpot)# [N x 1024 x 1024]
                spot_part_out = th.matmul(spot_part, w_WSItoSpot)  # [N x 8 x 1024]
                spot_part_out = th.reshape(spot_part_out, (spot_part_out.shape[0],spot_part_out.shape[1],
                                                           int(math.sqrt(spot_part_out.shape[2])), int(math.sqrt(spot_part_out.shape[2]))))  # [N x 8 x 32 x 32]
                spot_part_dist[...,i*part_width:(i+1)*part_width,j*part_width:(j+1)*part_width]=spot_part_out

                F_SpottoWSI = th.matmul(WSI_part_T, spot_part)  # [N x 1024 x 1024]
                w_SpottoWSI = self.fc_modulation_2(F_SpottoWSI)  # [N x 1024 x 1024]
                sfmax_module = nn.Softmax(dim=-1)
                w_SpottoWSI = sfmax_module(w_SpottoWSI)  # [N x 1024 x 1024]
                WSI_part_out = th.matmul(WSI_part, w_SpottoWSI)  # [N x 8 x 1024]
                WSI_part_out = th.reshape(WSI_part_out, (WSI_part_out.shape[0], WSI_part_out.shape[1], int(math.sqrt(WSI_part_out.shape[2])),
                                                           int(math.sqrt(WSI_part_out.shape[2]))))  # [N x 8 x 32 x 32]
                WSI_part_dist[..., i * part_width:(i + 1) * part_width,j * part_width:(j + 1) * part_width] = WSI_part_out
        ### weight
        WSI_part_dist = 0.9*dist_WSI+0.1*WSI_part_dist
        spot_part_dist =  0.9*dist_spot+0.1*spot_part_dist
        h_condition = th.cat([com_h, WSI_part_dist,spot_part_dist], dim=1) # [N x 24 x 64 x 64]

        #########  cross attention for embedding condition
        h_condition_pre = th.reshape(h_condition, (h_condition.shape[0], h_condition.shape[1], -1))
        h_condition_pre = th.transpose(h_condition_pre, 1, 2)  # [N x 4096 x 24]

        h_x_pre = th.reshape(h_x, (h_x.shape[0], h_x.shape[1], -1))
        h_x_pre = th.transpose(h_x_pre, 1, 2)  # [N x 4096 x 16]

        q = self.to_q_con(h_x_pre)  # [N x 4096 x 16]
        k = self.to_k_con(h_condition_pre)  # [N x 4096 x 16]
        v = self.to_v_con(h_condition_pre)  # [N x 4096 x 16]
        mid_atten = torch.matmul(q, th.transpose(k, 1, 2))

        scale = q.shape[2] ** -0.5
        mid_atten = mid_atten * scale
        sfmax = nn.Softmax(dim=-1)
        mid_atten = sfmax(mid_atten)  # [N x 4096 x 4096]
        Final_merge = torch.matmul(mid_atten, v)  # [N x 4096 x 16]
        Final_merge = th.transpose(Final_merge, 1, 2)  # [N x 16 x 4096 ]
        Final_merge = th.reshape(Final_merge, (Final_merge.shape[0], Final_merge.shape[1], h_x.shape[2], h_x.shape[3]))  # [N x 16 x 64 x64 ]

        #new1.5#t2
        Gene_index_map = Gene_index_map.float() 
        Gene_index_map1 = torch.nn.functional.interpolate(Gene_index_map, (64, 64)) # b, 10, 64, 64
        # Gene_index_map2 = torch.nn.functional.interpolate(Gene_index_map, (256, 256)).float()  # b, 10, 64, 64
        Final_merge = th.cat([Final_merge, Gene_index_map1], dim=1)
        
        h = self.conv_layer1(Final_merge)
        h = self.middle_block(h, emb)
        
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        #t1
        # Gene_index_map = self.conv_map(Gene_index_map)
        h = th.cat([h, Gene_index_map], dim=1)
        h = self.conv_layer2(h)
        #new1.5#t2

        return com_WSI, com_spot,  dist_WSI, dist_spot, GMSD_loss, self.out(h)

class GeneMemoryNetwork(nn.Module):
    def __init__(self, query_dim, num_genes, gene_dim, memory_bank):
        """
        :param query_dim: 单基因查询向量的通道数 (例如 3)，对应输入的查询特征维度
        :param num_genes: 基因记忆库中的基因数量 (例如 200)
        :param gene_dim: 每个基因的特征维度 (例如 32)
        :param memory_bank: 相关性矩阵，维度为 [num_genes, num_genes]，表示基因之间的相关性
        """
        super(GeneMemoryNetwork, self).__init__()
        self.query_dim = query_dim  # 输入查询向量的通道数
        self.num_genes = num_genes  # 基因数量
        self.gene_dim = gene_dim  # 基因特征维度
        self.memory_bank = memory_bank  # 相关性矩阵，维度为 [num_genes, num_genes]

        # 线性变换层 (用于注意力机制的 Query、Key 和 Value)
        self.WQ = nn.Linear(gene_dim, gene_dim)  # 查询向量线性变换
        self.WK = nn.Linear(gene_dim, gene_dim)  # 基因记忆库映射到 gene_dim
        self.WV = nn.Linear(gene_dim, gene_dim)  # 基因记忆库映射到 gene_dim


        # 多头注意力机制
        self.cross_attention = nn.MultiheadAttention(embed_dim=gene_dim, num_heads=8)

        # 还原特征通道到输入查询维度
        self.restore_channels = nn.Linear(gene_dim, 64*64)

    def forward(self, single_gene_data):
        """
        前向传播
        :param single_gene_data: 单基因查询数据 [batch, query_dim, 32, 32]
        :return: 优化后的单基因特征 [batch, query_dim, 32, 32]
        """
        batch_size, _, height, width = single_gene_data.shape
        queries = self.extract_queries(single_gene_data)  # [batch, query_dim, gene_dim]
        global_memory = self.extract_gene_features(self.memory_bank)  # [num_genes, gene_dim]

        query_proj = self.WQ(queries.cuda())
        key_proj = self.WK(global_memory)
        value_proj = self.WV(global_memory)
        query_proj = query_proj.permute(1, 0, 2)  # [query_dim, batch, gene_dim]

        key_proj = key_proj.permute(1, 0, 2)
        value_proj = value_proj.permute(1, 0, 2)
        attn_output, _ = self.cross_attention(query_proj, key_proj, value_proj)  # [query_dim, batch, gene_dim]
        attn_output = attn_output.permute(1, 0, 2)  # [batch, query_dim, gene_dim]

        # 4. 恢复原始通道维度
        restored = self.restore_channels(attn_output)  # [batch, query_dim, 32, 32]

        restored = restored.view(batch_size, self.query_dim, height, width)  # [batch, query_dim, 32, 32]

        return restored

    def extract_local_memory(self, single_gene_data):
        # 从当前 batch 提取局部记忆
        batch_memory = single_gene_data.mean(dim=[2, 3])  # [batch, query_dim]
        return batch_memory

    def extract_queries(self, single_gene_data):
        """
        提取单基因查询特征
        :param single_gene_data: [batch, query_dim, 32, 32]
        :return: 查询特征 [batch, query_dim, gene_dim]
        """
        batch_size, channels, height, width = single_gene_data.shape
        # 平均池化，将空间维度展平
        queries = single_gene_data.mean(dim=[2, 3])  # [batch, query_dim]
        return queries.unsqueeze(-1).expand(-1, -1, self.gene_dim)  # [batch, query_dim, gene_dim]
    

    
    def extract_gene_features(self, memory_bank):
        """
        提取基因记忆特征
        :param memory_bank: 相关性矩阵，维度为 [num_genes, num_genes]
        :return: 基因特征 [num_genes, gene_dim]
        """
        # 将 memory_bank 从 [num_genes, num_genes] 转换为 [num_genes, gene_dim]
        # 这里可以使用线性变换或其他处理方法
        # 为了示例，我采用平均池化的方式：根据相关性矩阵求平均值，映射到 gene_dim。
        if not isinstance(memory_bank, torch.Tensor):
            memory_bank = torch.tensor(memory_bank, dtype=torch.float32)
        gene_features = memory_bank.mean(dim=-1)  # [num_genes]
        # 然后扩展到 [num_genes, gene_dim]
        return gene_features.unsqueeze(-1).expand(-1, self.gene_dim)  # [num_genes, gene_dim]

class MetaMemoryUpdater(nn.Module):
    def __init__(self, gene_dim):
        """
        元记忆调整模块
        :param gene_dim: 基因特征维度
        """
        super().__init__()
        self.adaptive_layer = nn.Linear(gene_dim, gene_dim)  # 用于动态调整记忆库

    def adapt_memory(self, memory_bank, query_features):
        """
        根据查询特征动态调整记忆库
        :param memory_bank: 原始全局记忆库 [num_genes, gene_dim]
        :param query_features: 查询特征 [batch, query_dim, gene_dim]
        :return: 调整后的记忆库 [num_genes, gene_dim]
        """
        task_adaptation = query_features.mean(dim=0)  # 计算任务特征 [query_dim, gene_dim]
        adapted_memory = memory_bank.to('cuda') + self.adaptive_layer(task_adaptation.mean(dim=0)).to('cuda')  # 动态调整
        return adapted_memory


class SparseAttention(nn.Module):
    def __init__(self, sparsity=0.1):
        """
        稀疏注意力模块
        :param sparsity: 稀疏率，表示保留的记忆比例 (0.1 表示保留 10%)
        """
        super().__init__()
        self.sparsity = sparsity

    def forward(self, query, memory):
        """
        稀疏选择与加权
        :param query: 查询向量 [batch, query_dim, gene_dim]
        :param memory: 记忆库 [num_genes, gene_dim]
        :return: 稀疏加权的记忆特征 [batch, query_dim, gene_dim]
        """
        # 计算相似性分数
        scores = torch.matmul(query, memory.T)  # [batch, query_dim, num_genes]

        # Top-K 筛选
        topk_scores, topk_indices = torch.topk(scores, int(memory.size(0) * self.sparsity), dim=-1)  # [batch, query_dim, top_k]

        # 提取稀疏记忆
        batch_size, query_dim, top_k = topk_scores.shape
        gene_dim = memory.size(-1)

        # Flatten topk_indices for gathering memory
        sparse_memory = memory[topk_indices.view(-1)]  # [batch * query_dim * top_k, gene_dim]

        # Reshape sparse_memory to restore batch and query dimensions
        sparse_memory = sparse_memory.view(batch_size, query_dim, top_k, gene_dim)  # [batch, query_dim, top_k, gene_dim]

        # Softmax on scores to generate weights
        sparse_weights = torch.softmax(topk_scores, dim=-1)  # [batch, query_dim, top_k]

        # 加权求和
        # 注意：sparse_weights 与 sparse_memory 的形状需要匹配
        weighted_memory = torch.einsum('bqt,bqtd->bqd', sparse_weights, sparse_memory)  # [batch, query_dim, gene_dim]

        return weighted_memory

class FeatureNoiseReplacer:
    def __init__(self, replacement_prob=0.8):
        """
        初始化替换器
        :param replacement_prob: 替换为噪声的概率，默认80%
        """
        self.replacement_prob = replacement_prob

    def replace_with_noise(self, feature_map, ratio, dtype=torch.float32):
        """
        根据比例替换特征为噪声
        :param feature_map: 输入特征张量，形状为 [batch, channels, height, width]
        :param ratio: 替换为噪声的比率，值在 0 到 1 之间
        :param dtype: 数据类型
        :return: 替换后的特征张量
        """
        #new1.5
        if ratio == 2.0:
            ratio = 0.2
        elif ratio <0.5:
            ratio = 0.0
        else:
            ratio = ratio / 4.0
        #new1.5
        # 生成与特征张量形状相同的随机噪声
        noise = torch.randn_like(feature_map, dtype=dtype)

        # 计算需要替换的总元素数
        total_elements = feature_map.numel()
        num_replace = int(total_elements * ratio) # 计算需要替换的数量

        # 展平特征图以随机选择需要替换的元素
        flat_indices = torch.randperm(total_elements, device=feature_map.device)[:num_replace]

        # 创建一个掩码，将选中的位置设置为 True
        mask = torch.zeros(total_elements, device=feature_map.device, dtype=torch.bool)
        mask[flat_indices] = True

        # 将掩码 reshape 回到原始特征图的形状
        mask = mask.view_as(feature_map)

        # 使用掩码进行条件替换
        replaced_feature = torch.where(mask, noise, feature_map)
        return replaced_feature


