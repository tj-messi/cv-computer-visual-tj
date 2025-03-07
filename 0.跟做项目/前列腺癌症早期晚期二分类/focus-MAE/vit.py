import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#adamae
from functools import partial
import cv2 
import numpy as np
# from models.finetune.vit import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from .modeling_finetune import (
    Block,
    PatchEmbed,
    _cfg,
    get_sinusoid_encoding_table,
)
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

# only import following models
__all__ = [
    'pretrain_adamae_base_patch16_224', 
    'pretrain_adamae_large_patch16_224',
]

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,
                 use_learnable_pos_emb=False, mask_ratio=0.95, bcos=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.visible_patches = int(num_patches*(1-mask_ratio))
        print("No. of visible patches selected for pre-training: {}".format(self.visible_patches))

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, bcos_attn=bcos)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Probability prediction network
        self.pos_embed_probs = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.get_token_probs = nn.Sequential(
                                Block(dim=embed_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                                drop=0.1, attn_drop=0.00, drop_path=0.00, norm_layer=nn.LayerNorm,
                                init_values=0.,bcos_attn=bcos),
                                nn.Linear(embed_dim, 1),
                                torch.nn.Flatten(start_dim=1),
                                )
                            
        self.softmax =  nn.Softmax(dim=-1)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def get_mask(self, x, priors, prior_mask,delta):
        x = x + self.pos_embed_probs.type_as(x).to(x.device).clone() #detach()
        # if priors:
        #     logits = self.get_token_probs(x[~priors])
        # else:
        logits = self.get_token_probs(x)
        logits =  torch.nan_to_num(logits)
        helper_tensor = torch.zeros(logits.shape).to(logits.device)
        helper_tensor[priors] = delta
        # print("logits ", logits.shape, torch.mean(logits), torch.max(logits), torch.min(logits))
        logits = logits + helper_tensor
        # print("logits ", logits.shape, logits, torch.mean(logits))
        
        p_x = self.softmax(logits)
        
        # print("p_x ", p_x.shape, torch.sum(p_x))
        
        vis_idx = torch.multinomial(p_x, num_samples=self.visible_patches, replacement=False)
        
        # priors = ~priors 
        prior_mask = False

        # print("vis_idx ", vis_idx.shape, torch.sum(vis_idx))
        if not prior_mask:
            mask = torch.ones((x.shape[0], x.shape[1]))#.to(priors.device)
            # mask = ~priors.int()

            mask = mask.to(x.device, non_blocking=True)

            mask.scatter_(dim=-1, index=vis_idx.long(), value=0.0) #0.0
            mask = mask.flatten(1).to(torch.bool)
            
        else:
            mask = priors
        # mask = mask | priors
        
        # test_img = mask.cpu().numpy().reshape(mask.shape[0] , 14, 14, 8)

       
        # import matplotlib.pyplot as plt

        # Assuming your matrix is named 'matrix' and has dimensions 8x14x14x8
        # You may need to adapt this code based on the structure of your data

        # Create a big blank canvas to place the images

        # canvas_size = (14 * 8, 14 * 8)
        # canvas = np.zeros(canvas_size, dtype=np.uint8)

        # # Iterate through the matrix and place each 14x14 image on the canvas
        # for i in range(mask.shape[0]):
        #     for j in range(8):
        #         image = test_img[i, :, :, j]*255  # Assuming the last dimension is the channel dimension
        #         x_start, y_start = i * 14, j * 14
        #         canvas[x_start:x_start + 14, y_start:y_start + 14] = image
        
        # cv2.imwrite("test_img.png", canvas)

        # mask = torch.ones(priors.shape).to(priors.device) -  priors.to(dtype=torch.int32)
        
       
        # mask = mask | priors


        # print("removed priors and adaptive mask shape and masked patches", mask.shape, torch.sum(mask)," prior shape" ,priors.shape,torch.sum(priors))
        return p_x, vis_idx, mask

    def forward_features(self, x, priors=None, prior_mask=False, delta=1.0):
        _, _, T, _, _ = x.shape #8, 3, 16(T), 224, 224
        x = self.patch_embed(x) #8, 1568 (224/16 x 224/16 x 16/2), 768
        p_x, vis_idx, mask = self.get_mask(x, priors, prior_mask, delta)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        B, _, C = x.shape
        

        p_x, vis_idx, mask = self.get_mask(x, priors, prior_mask, delta=delta)

        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible shape: 8, 160, 768

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis) #8, 160, 768
        return x_vis, p_x, vis_idx, mask

    def forward(self, x, priors= None, prior_mask=False, delta=1.0):
        x, p_x, vis_idx, mask = self.forward_features(x, priors=priors, prior_mask=prior_mask, delta=delta)
        x = self.head(x)
        return x, p_x, mask

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2,bcos=False
                 ):
        super().__init__()
        self.num_classes = num_classes
        # assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values,bcos_attn=bcos)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=2, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=1536, #  decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 mask_ratio=0.9,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 bcos=False,
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            mask_ratio=mask_ratio,
            bcos=bcos)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            bcos = bcos)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, priors, prior_mask, delta=1.0):
        _, _, T, _, _ = x.shape
        x_vis, p_x, mask = self.encoder(x , priors= priors, prior_mask=prior_mask,delta=delta) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        # try:        
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        # except:
        # #     print("masking error")
        #     pos_emd_vis = expand_pos_embed
        #     pos_emd_mask = None
        # except:
            
        #     mask = torch.ones(expand_pos_embed.shape[0], expand_pos_embed.shape[1]).to(expand_pos_embed.device)
        #     mask[: , 0::3] = 0.0
        #     mask = mask.bool()
        #     pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        #     pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)

        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x_mask = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16] pos_emd_mask.shape[1]

        return x_mask, p_x, mask

class Pretrain_VisionTransformer_Encoder_Gating(nn.Module):
    '''
    2025-3-4 zjz
    Masking Probability NetWork : Gating Network
    '''
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,
                 use_learnable_pos_emb=False, mask_ratio=0.95, bcos=False):
        super().__init__()
        self.num_segments = 3
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.visible_patches = int(num_patches*(1-mask_ratio))
        print("No. of visible patches selected for pre-training: {}".format(self.visible_patches))

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer,
                init_values=init_values,
                bcos_attn=bcos)for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Probability prediction network
        '''
        self.pos_embed_probs = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.get_token_probs = nn.Sequential(
                                Block(dim=embed_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                                drop=0.1, attn_drop=0.00, drop_path=0.00, norm_layer=nn.LayerNorm,
                                init_values=0.,bcos_attn=bcos),
                                nn.Linear(embed_dim, 1),
                                torch.nn.Flatten(start_dim=1),
                                )
                            
        self.softmax =  nn.Softmax(dim=-1)
        '''

        # Gating Network
        self.pos_embed_probs = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.get_token_probs = nn.Sequential(
                                Block(dim=embed_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                                drop=0.1, attn_drop=0.00, drop_path=0.00, norm_layer=nn.LayerNorm,
                                init_values=0.,bcos_attn=bcos),
                                nn.Linear(embed_dim, 1),
                                torch.nn.Flatten(start_dim=1),
                                )
        self.delta_block = nn.Parameter(torch.ones(num_patches, self.num_segments))

        '''
        self.num_features = self.embed_dim = embed_dim
        '''
        self.softmax =  nn.Softmax(dim=-1)



        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def get_mask(self, x, priors, prior_mask,segment_masks,delta):
        x = x + self.pos_embed_probs.type_as(x).to(x.device).clone() #detach()
        logits = self.get_token_probs(x)
        logits =  torch.nan_to_num(logits)
        helper_tensor = torch.zeros(logits.shape).to(logits.device)

        adjusted_delta = self.delta_block * segment_masks.sum(dim=-1)

        helper_tensor[:] = adjusted_delta
        # print("logits ", logits.shape, torch.mean(logits), torch.max(logits), torch.min(logits))
        logits = logits + helper_tensor
        # print("logits ", logits.shape, logits, torch.mean(logits))
        
        p_x = self.softmax(logits)
        
        # print("p_x ", p_x.shape, torch.sum(p_x))
        
        vis_idx = torch.multinomial(p_x, num_samples=self.visible_patches, replacement=False)
        
        # priors = ~priors 
        prior_mask = False

        # print("vis_idx ", vis_idx.shape, torch.sum(vis_idx))
        if not prior_mask:
            mask = torch.ones((x.shape[0], x.shape[1]))#.to(priors.device)
            # mask = ~priors.int()

            mask = mask.to(x.device, non_blocking=True)

            mask.scatter_(dim=-1, index=vis_idx.long(), value=0.0) #0.0
            mask = mask.flatten(1).to(torch.bool)
            
        else:
            mask = priors
        return p_x, vis_idx, mask

    def forward_features(self, x, priors=None, prior_mask=False,segment_masks=None, delta=1.0):
        _, _, T, _, _ = x.shape #8, 3, 16(T), 224, 224
        x = self.patch_embed(x) #8, 1568 (224/16 x 224/16 x 16/2), 768
        p_x, vis_idx, mask = self.get_mask(x, priors, prior_mask, segment_masks,delta)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        B, _, C = x.shape
        

        p_x, vis_idx, mask = self.get_mask(x, priors, prior_mask, segment_masks,delta=delta)

        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible shape: 8, 160, 768

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis) #8, 160, 768
        return x_vis, p_x, vis_idx, mask

    def forward(self, x, priors= None, prior_mask=False, segment_masks=None,delta=1.0):
        x, p_x, vis_idx, mask = self.forward_features(x, priors=priors, prior_mask=prior_mask,segment_masks=segment_masks, delta=delta)
        x = self.head(x)
        return x, p_x, mask
class Pretrain_VisionTransformer_Decoder_Gating(nn.Module):
    """ 
    2025-3-4 zjz
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2,bcos=False
                 ):
        super().__init__()
        self.num_classes = num_classes
        # assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values,bcos_attn=bcos)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x
class Pretrain_VisionTransformer_Gating(nn.Module):
    """ 
    2025-3-4 zjz Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=1536, #  decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 mask_ratio=0.9,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 bcos=False,
                 ):
        super().__init__()
        self.encoder = Pretrain_VisionTransformer_Encoder_Gating(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            mask_ratio=mask_ratio,
            bcos=bcos)

        self.decoder = Pretrain_VisionTransformer_Decoder_Gating(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            bcos = bcos)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, priors, prior_mask, segment_masks, delta=1.0):
        _, _, T, _, _ = x.shape
        x_vis, p_x, mask = self.encoder(x , priors= priors, prior_mask=prior_mask,segment_masks=segment_masks,delta=delta) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        # try:        
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        # except:
        # #     print("masking error")
        #     pos_emd_vis = expand_pos_embed
        #     pos_emd_mask = None
        # except:
            
        #     mask = torch.ones(expand_pos_embed.shape[0], expand_pos_embed.shape[1]).to(expand_pos_embed.device)
        #     mask[: , 0::3] = 0.0
        #     mask = mask.bool()
        #     pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        #     pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)

        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x_mask = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16] pos_emd_mask.shape[1]

        return x_mask, p_x, mask



@register_model
def pretrain_adamae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=192, 
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mask_ratio=0.9,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
        


@register_model
def pretrain_adamae_base_bcos_patch16_224(pretrained=False, mask_ratio=0.9, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        mask_ratio=mask_ratio,
        bcos=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_adamae_base_patch16_224(pretrained=False, mask_ratio=0.9, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=384, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        mask_ratio=mask_ratio,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 
@register_model
def pretrain_adamae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        mask_ratio=0.9,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_focusmae_small_patch_base_model(pretrained=False, mask_ratio=0.9, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=14, 
        encoder_embed_dim=384, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1176,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        mask_ratio=mask_ratio,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model