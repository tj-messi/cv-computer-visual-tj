#finetune部分的分割利用

##初步方法

利用分割作为重要性乘以第一个x映射后的特征

	 def forward_features(self, x ,mask):
	        Batch_size = mask.size(0)
	        B = x.size(0)
	
	        x = self.patch_embed(x)
	
	        embedded_x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
	        # x : batch_size,1568(14x14x16/2),768
	        # 乘上重要性
	        self.encoder_mask_map_generator = SegmentMaskingGenerator((16, 224, 224),Segments=mask)
	        encoder_mask_map = self.encoder_mask_map_generator()
	        import_x = self.MoE(embedded_x,encoder_mask_map)
	        import_x = torch.sigmoid(import_x)  # shape (1, 1568)
	        # print(
	        #     "min:", import_x.min().item(),
	        #     "max:", import_x.max().item(),
	        #     "mean:", import_x.mean().item(),
	        #     "std:", import_x.std().item(),
	        #     "sum:", import_x.sum().item()
	        # )
	
	        import_x = import_x.unsqueeze(-1)
	        x = x * import_x
	
	        if self.pos_embed is not None:
	            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(
	                x.device).clone().detach()
	        x = self.pos_drop(x)
	
	        for blk in self.blocks:
	            if self.with_cp:
	                x = cp.checkpoint(blk, x)
	            else:
	                x = blk(x)
	
	        if self.fc_norm is not None:
	            return self.fc_norm(x.mean(1))
	        else:
	            return self.norm(x[:, 0])
	        
	        # 返回聚合信息(3,768)

##改进方法 FocusMAE

	    def forward_features(self, x ,mask):
	        Batch_size = mask.size(0)
	        B = x.size(0)
	
	        x = self.patch_embed(x)
	
	        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
	        # x : batch_size,1568(14x14x16/2),768
	        # 乘上重要性
	        self.encoder_mask_map_generator = SegmentMaskingGenerator((16, 224, 224),Segments=mask)
	        encoder_mask_map = self.encoder_mask_map_generator()
	        score_patch = self.MoE(x,encoder_mask_map)
	
	        import_x = torch.softmax(score_patch, dim=1)  # (batch_size, 1568)
	        vis_idx = torch.multinomial(import_x, num_samples=int(1568*0.9), replacement=False)
	        masked = torch.ones((x.shape[0], x.shape[1]))#.to(priors.device)
	        masked = masked.to(x.device, non_blocking=True)
	        masked.scatter_(dim=-1, index=vis_idx.long(), value=0.0) #0.0
	        masked = masked.flatten(1).to(torch.bool)
	
	        B, _, C = x.shape
	        x_vis = x[~masked].reshape(B, -1, C)
	
	        x_vis = self.pos_drop(x_vis)
	
	        for blk in self.blocks:
	            if self.with_cp:
	                x_vis = cp.checkpoint(blk, x_vis)
	            else:
	                x_vis = blk(x_vis)
	
	        if self.fc_norm is not None:
	            return self.fc_norm(x_vis.mean(1))
	        else:
	            return self.norm(x_vis[:, 0])
	        
	        # 返回聚合信息(3,768)

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1743600775741.png)


#topk做额外Prompt

	def forward_features(self, x ,mask):
		B = x.size(0)

        x = self.patch_embed(x)

        B, N, C = x.shape

        embedded_x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        self.encoder_mask_map_generator = SegmentMaskingGenerator((16, 224, 224),Segments=mask)
        encoder_mask_map = self.encoder_mask_map_generator()
        import_x = self.MoE(embedded_x,encoder_mask_map)
        import_x = torch.sigmoid(import_x)  # shape (1, 1568)

        topk_ratio = 0.1  # 取 10%
        topk_num = max(1, int(N * topk_ratio))

        _, topk_indices = torch.topk(import_x, topk_num, dim=1)  # shape (B, topk_num)
        prompt_tokens = torch.gather(x, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, C))  # shape (B, topk_num, C)

        # 将 Prompt 拼接到输入
        x = torch.cat([prompt_tokens, x], dim=1)  # shape (B, N + topk_num, C)

        # 位置编码同步扩展
        if self.pos_embed is not None:
            pos_embed_expanded = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
            pos_prompt = torch.gather(pos_embed_expanded, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, C))
            x = x + torch.cat([pos_prompt, pos_embed_expanded], dim=1)

        x = self.pos_drop(x)

        # 经过 Transformer 编码
        for blk in self.blocks:
            if self.with_cp:
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)

        # 输出分类特征
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return self.norm(x[:, 0])

#clip数量问题

可以按照论文里面这样说的：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1743734586518.png)

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1743734600723.png)


#多尺度注意力

	class VisionTransformer(nn.Module):
	    """ Vision Transformer with support for patch or hybrid CNN input stage
	    """
	
	    def __init__(self,
	                 img_size=224,
	                 patch_size=16,
	                 in_chans=3,
	                 num_classes=2,
	                 embed_dim=768,
	                 depth=12,
	                 num_heads=12,
	                 mlp_ratio=4.,
	                 qkv_bias=False,
	                 qk_scale=None,
	                 drop_rate=0.,
	                 attn_drop_rate=0.,
	                 drop_path_rate=0.,
	                 head_drop_rate=0.,
	                 norm_layer=nn.LayerNorm,
	                 init_values=0.,
	                 use_learnable_pos_emb=False,
	                 init_scale=0.,
	                 all_frames=16,
	                 tubelet_size=2,
	                 use_mean_pooling=True,
	                 with_cp=False,
	                 cos_attn=False,
	                 fusion_layers=[3, 6, 9, 12]):
	        super().__init__()
	
	        self.fusion_layers = fusion_layers  # 记录要融合的层索引
	        
	        # 添加特征融合相关层
	        self.fusion_projections = nn.ModuleList([
	            nn.Linear(embed_dim, embed_dim) for _ in fusion_layers
	        ])
	        self.fusion_norm = norm_layer(embed_dim)
	        self.num_classes = num_classes
	
	        # num_features for consistency with other models
	        self.num_features = self.embed_dim = embed_dim
	        self.tubelet_size = tubelet_size
	        self.patch_embed = PatchEmbed(
	            img_size=img_size,
	            patch_size=patch_size,
	            in_chans=in_chans,
	            embed_dim=embed_dim,
	            num_frames=all_frames,
	            tubelet_size=tubelet_size)
	        num_patches = self.patch_embed.num_patches
	        self.with_cp = with_cp
	
	        if use_learnable_pos_emb:
	            self.pos_embed = nn.Parameter(
	                torch.zeros(1, num_patches, embed_dim))
	        else:
	            # sine-cosine positional embeddings is on the way
	            self.pos_embed = get_sinusoid_encoding_table(
	                num_patches, embed_dim)
	
	        self.pos_drop = nn.Dropout(p=drop_rate)
	
	        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
	               ]  # stochastic depth decay rule
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
	                cos_attn=cos_attn) for i in range(depth)
	        ])
	        self.norm = nn.Identity() if use_mean_pooling else norm_layer(
	            embed_dim)
	        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
	        self.head_dropout = nn.Dropout(head_drop_rate)
	        self.head = nn.Linear(
	            embed_dim, num_classes) if num_classes > 0 else nn.Identity()
	
	        if use_learnable_pos_emb:
	            trunc_normal_(self.pos_embed, std=.02)
	
	        self.apply(self._init_weights)
	
	        self.head.weight.data.mul_(init_scale)
	        self.head.bias.data.mul_(init_scale)
	
	        self.MoE = MoE(input_dim=embed_dim, num_experts=3)
	        self.softmax =  nn.Softmax(dim=-1)
	
	        for param in self.MoE.parameters():
	            param.requires_grad = False
	
	    def _init_weights(self, m):
	        if isinstance(m, nn.Linear):
	            trunc_normal_(m.weight, std=.02)
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
	
	    def forward_features(self, x ,mask):
	        B = x.size(0)
	
	        x = self.patch_embed(x)
	
	        B, N, C = x.shape
	
	        embedded_x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
	        self.encoder_mask_map_generator = SegmentMaskingGenerator((16, 224, 224),Segments=mask)
	        encoder_mask_map = self.encoder_mask_map_generator()
	        import_x = self.MoE(embedded_x,encoder_mask_map)
	        import_x = torch.sigmoid(import_x)  # shape (1, 1568)
	
	        topk_ratio = 0.1  # 取 10%
	        topk_num = max(1, int(N * topk_ratio))
	
	        _, topk_indices = torch.topk(import_x, topk_num, dim=1)  # shape (B, topk_num)
	        prompt_tokens = torch.gather(x, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, C))  # shape (B, topk_num, C)
	
	        # 将 Prompt 拼接到输入
	        x = torch.cat([prompt_tokens, x], dim=1)  # shape (B, N + topk_num, C)
	
	        # 位置编码同步扩展
	        if self.pos_embed is not None:
	            pos_embed_expanded = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
	            pos_prompt = torch.gather(pos_embed_expanded, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, C))
	            x = x + torch.cat([pos_prompt, pos_embed_expanded], dim=1)
	
	        x = self.pos_drop(x)
	
	        # 存储多尺度特征
	        multi_scale_features = []
	        
	        # 通过 Transformer blocks 并收集指定层的特征
	        for i, blk in enumerate(self.blocks):
	            if self.with_cp:
	                x = cp.checkpoint(blk, x)
	            else:
	                x = blk(x)
	                
	            # 如果当前层在 fusion_layers 中，保存特征
	            if (i + 1) in self.fusion_layers:
	                if self.fc_norm is not None:
	                    feat = self.fc_norm(x.mean(1))
	                else:
	                    feat = self.norm(x[:, 0])
	                multi_scale_features.append(feat)
	
	        # 特征融合
	        if len(multi_scale_features) > 1:
	            # 对每个尺度的特征进行投影
	            projected_features = [
	                proj(feat) for proj, feat in zip(self.fusion_projections, multi_scale_features)
	            ]
	            # 堆叠并融合（这里使用简单平均，也可以使用加权平均或其他融合方式）
	            fused_features = torch.stack(projected_features, dim=0).mean(dim=0)
	            # 应用归一化
	            fused_features = self.fusion_norm(fused_features)
	            return fused_features
	        else:
	            # 如果只有一个特征层，直接返回
	            return multi_scale_features[0]
	
	    def forward(self, x , mask):
	        # x : 3 3 16 224 224 batch_size channel num_frames height weight
	        # mask : torch.Size([1, 16, 3, 3, 224, 224])
	        mask = mask.squeeze(0)  # shape: [16, 3, 3, 224, 224]
	        x = self.forward_features(x,mask)
	        x = self.head_dropout(x)
	        x = self.head(x)
	        # x.shape:3,2
	        return x

#可学习的多尺度特征

	class VisionTransformer(nn.Module):
	    """ Vision Transformer with support for patch or hybrid CNN input stage
	    """
	
	    def __init__(self,
	                 img_size=224,
	                 patch_size=16,
	                 in_chans=3,
	                 num_classes=2,
	                 embed_dim=768,
	                 depth=12,
	                 num_heads=12,
	                 mlp_ratio=4.,
	                 qkv_bias=False,
	                 qk_scale=None,
	                 drop_rate=0.,
	                 attn_drop_rate=0.,
	                 drop_path_rate=0.,
	                 head_drop_rate=0.,
	                 norm_layer=nn.LayerNorm,
	                 init_values=0.,
	                 use_learnable_pos_emb=False,
	                 init_scale=0.,
	                 all_frames=16,
	                 tubelet_size=2,
	                 use_mean_pooling=True,
	                 with_cp=False,
	                 cos_attn=False,
	                 fusion_layers=[3, 6, 9, 12]):
	        super().__init__()
	
	        self.fusion_layers = fusion_layers  # Layers to be fused
	        
	        # Adding feature fusion related layers
	        self.fusion_projections = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in fusion_layers])
	        self.fusion_norm = norm_layer(embed_dim)
	        
	        # Attention-based fusion mechanism
	        self.attn_fusion = nn.MultiheadAttention(embed_dim, num_heads=6)  # Single-head attention for fusion
	
	        # num_features for consistency with other models
	        self.num_features = self.embed_dim = embed_dim
	        self.tubelet_size = tubelet_size
	        self.patch_embed = PatchEmbed(
	            img_size=img_size,
	            patch_size=patch_size,
	            in_chans=in_chans,
	            embed_dim=embed_dim,
	            num_frames=all_frames,
	            tubelet_size=tubelet_size)
	        num_patches = self.patch_embed.num_patches
	        self.with_cp = with_cp
	
	        if use_learnable_pos_emb:
	            self.pos_embed = nn.Parameter(
	                torch.zeros(1, num_patches, embed_dim))
	        else:
	            # sine-cosine positional embeddings is on the way
	            self.pos_embed = get_sinusoid_encoding_table(
	                num_patches, embed_dim)
	
	        self.pos_drop = nn.Dropout(p=drop_rate)
	
	        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
	               ]  # stochastic depth decay rule
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
	                cos_attn=cos_attn) for i in range(depth)
	        ])
	        self.norm = nn.Identity() if use_mean_pooling else norm_layer(
	            embed_dim)
	        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
	        self.head_dropout = nn.Dropout(head_drop_rate)
	        self.head = nn.Linear(
	            embed_dim, num_classes) if num_classes > 0 else nn.Identity()
	
	        if use_learnable_pos_emb:
	            trunc_normal_(self.pos_embed, std=.02)
	
	        self.apply(self._init_weights)
	
	        self.head.weight.data.mul_(init_scale)
	        self.head.bias.data.mul_(init_scale)
	
	        self.MoE = MoE(input_dim=embed_dim, num_experts=3)
	        self.softmax =  nn.Softmax(dim=-1)
	
	        for param in self.MoE.parameters():
	            param.requires_grad = False
	
	    def _init_weights(self, m):
	        if isinstance(m, nn.Linear):
	            trunc_normal_(m.weight, std=.02)
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
	
	    def forward_features(self, x ,mask):
	        B = x.size(0)
	
	        # Patch embedding + position encoding
	        x = self.patch_embed(x)
	        B, N, C = x.shape
	        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
	        x = self.pos_drop(x)
	
	        # Multi-scale feature collection
	        multi_scale_features = []
	
	        for i, blk in enumerate(self.blocks):
	            if self.with_cp:
	                x = cp.checkpoint(blk, x)
	            else:
	                x = blk(x)
	                
	            if (i + 1) in self.fusion_layers:
	                if self.fc_norm is not None:
	                    feat = self.fc_norm(x.mean(1))  # Mean pooling across patches
	                else:
	                    feat = self.norm(x[:, 0])  # Taking the cls token as feature
	                multi_scale_features.append(feat)
	
	        # Attention-based fusion
	        if len(multi_scale_features) > 1:
	            # Project features to the same space
	            projected_features = [proj(feat) for proj, feat in zip(self.fusion_projections, multi_scale_features)]
	            projected_features = torch.stack(projected_features, dim=0)  # Shape: [num_layers, B, embed_dim]
	
	            # Apply attention for fusion: projected_features will be used as query, key, and value
	            fused_features, _ = self.attn_fusion(projected_features, projected_features, projected_features)
	            fused_features = fused_features.mean(dim=0)  # Average across layers
	
	            # Apply normalization
	            fused_features = self.fusion_norm(fused_features)
	            return fused_features
	        else:
            return multi_scale_features[0]


#3,6,9层平均+concat给12层多尺度特征

	class VisionTransformer(nn.Module):
	    """ Vision Transformer with support for patch or hybrid CNN input stage
	    """
	
	    def __init__(self,
	                 img_size=224,
	                 patch_size=16,
	                 in_chans=3,
	                 num_classes=2,
	                 embed_dim=768,
	                 depth=12,
	                 num_heads=12,
	                 mlp_ratio=4.,
	                 qkv_bias=False,
	                 qk_scale=None,
	                 drop_rate=0.,
	                 attn_drop_rate=0.,
	                 drop_path_rate=0.,
	                 head_drop_rate=0.,
	                 norm_layer=nn.LayerNorm,
	                 init_values=0.,
	                 use_learnable_pos_emb=False,
	                 init_scale=0.,
	                 all_frames=16,
	                 tubelet_size=2,
	                 use_mean_pooling=True,
	                 with_cp=False,
	                 cos_attn=False,
	                 fusion_layers=[3, 6, 9, 12],
	                 average_layers=[3,6,9]):
	        super().__init__()
	        attn_drop_rate = 0.3
	        drop_path_rate = 0.3
	        head_drop_rate = 0.3
	        
	        self.fusion_layers = fusion_layers  # 记录要融合的层索引
	        self.average_layers = average_layers
	
	        # 添加特征融合相关层
	        self.fusion_projections = nn.ModuleList([
	            nn.Linear(embed_dim, embed_dim) for _ in fusion_layers
	        ])
	        self.fusion_norm = norm_layer(embed_dim)
	        self.num_classes = num_classes
	
	        # 1,1536 -> 1,768
	        self.conv_layer = nn.Conv1d(in_channels=1536, out_channels=768, kernel_size=1)
	
	        # num_features for consistency with other models
	        self.num_features = self.embed_dim = embed_dim
	        self.tubelet_size = tubelet_size
	        self.patch_embed = PatchEmbed(
	            img_size=img_size,
	            patch_size=patch_size,
	            in_chans=in_chans,
	            embed_dim=embed_dim,
	            num_frames=all_frames,
	            tubelet_size=tubelet_size)
	        num_patches = self.patch_embed.num_patches
	        self.with_cp = with_cp
	
	        if use_learnable_pos_emb:
	            self.pos_embed = nn.Parameter(
	                torch.zeros(1, num_patches, embed_dim))
	        else:
	            # sine-cosine positional embeddings is on the way
	            self.pos_embed = get_sinusoid_encoding_table(
	                num_patches, embed_dim)
	
	        self.pos_drop = nn.Dropout(p=drop_rate)
	
	        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
	               ]  # stochastic depth decay rule
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
	                cos_attn=cos_attn) for i in range(depth)
	        ])
	        self.norm = nn.Identity() if use_mean_pooling else norm_layer(
	            embed_dim)
	        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
	        self.head_dropout = nn.Dropout(head_drop_rate)
	        self.head = nn.Linear(
	            embed_dim, num_classes) if num_classes > 0 else nn.Identity()
	
	        if use_learnable_pos_emb:
	            trunc_normal_(self.pos_embed, std=.02)
	
	        self.apply(self._init_weights)
	
	        self.head.weight.data.mul_(init_scale)
	        self.head.bias.data.mul_(init_scale)
	
	        self.MoE = MoE(input_dim=embed_dim, num_experts=3)
	        self.softmax =  nn.Softmax(dim=-1)
	
	        for param in self.MoE.parameters():
	            param.requires_grad = False
	
	    def _init_weights(self, m):
	        if isinstance(m, nn.Linear):
	            trunc_normal_(m.weight, std=.02)
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
	
	    def forward_features(self, x ,mask):
	        B = x.size(0)
	
	        x = self.patch_embed(x)
	
	        B, N, C = x.shape
	
	        embedded_x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
	        self.encoder_mask_map_generator = SegmentMaskingGenerator((16, 224, 224),Segments=mask)
	        encoder_mask_map = self.encoder_mask_map_generator()
	        import_x = self.MoE(embedded_x,encoder_mask_map)
	        import_x = torch.sigmoid(import_x)  # shape (1, 1568)
	
	        topk_ratio = 0.1  # 取 10%
	        topk_num = max(1, int(N * topk_ratio))
	
	        _, topk_indices = torch.topk(import_x, topk_num, dim=1)  # shape (B, topk_num)
	        prompt_tokens = torch.gather(x, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, C))  # shape (B, topk_num, C)
	
	        # 将 Prompt 拼接到输入
	        x = torch.cat([prompt_tokens, x], dim=1)  # shape (B, N + topk_num, C)
	
	        # 位置编码同步扩展
	        if self.pos_embed is not None:
	            pos_embed_expanded = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
	            pos_prompt = torch.gather(pos_embed_expanded, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, C))
	            x = x + torch.cat([pos_prompt, pos_embed_expanded], dim=1)
	
	        x = self.pos_drop(x)
	
		
	        # 存储多尺度特征
	        multi_scale_features = []
	        layer12_feature = None
	
	        # 通过 Transformer blocks 并收集指定层的特征
	        for i, blk in enumerate(self.blocks):
	            if self.with_cp:
	                x = cp.checkpoint(blk, x)
	            else:
	                x = blk(x)
	                
	            # 收集 3, 6, 9 层的特征用于平均
	            if (i + 1) in self.average_layers:  # [3, 6, 9]
	                if self.fc_norm is not None:
	                    feat = self.fc_norm(x.mean(1))
	                else:
	                    feat = self.norm(x[:, 0])
	                multi_scale_features.append(feat)
	            
	            # 单独保存第 12 层特征
	            if (i + 1) == 12:
	                if self.fc_norm is not None:
	                    layer12_feature = self.fc_norm(x.mean(1))
	                else:
	                    layer12_feature = self.norm(x[:, 0])
	
	        # 计算 3, 6, 9 层特征的平均值
	        if multi_scale_features:
	            avg_features = torch.stack(multi_scale_features, dim=0).mean(dim=0)  # Average features from layers 3, 6, 9
	            # Use the first projection layer for the averaged feature
	            avg_projected = self.fusion_projections[0](avg_features)  # Project averaged feature
	            avg_projected = self.fusion_norm(avg_projected)
	        else:
	            raise ValueError("No features collected from layers 3, 6, 9")
	
	        # 12 层特征
	        if layer12_feature is not None:
	            # Use the last projection layer for layer 12
	            layer12_projected = self.fusion_projections[-1](layer12_feature)  # Project layer 12 feature
	            layer12_projected = self.fusion_norm(layer12_projected)
	        else:
	            raise ValueError("Layer 12 feature not found")
	
	        # 拼接投影后的特征
	        fused_features = torch.cat([avg_projected, layer12_projected], dim=-1)
	        fused_features = fused_features.unsqueeze(-1)  #  (1, 1536, 1)
	        output_features = self.conv_layer(fused_features)
	        # (1, 768)
	        output_features = output_features.squeeze(-1)  #  (1, 768)
	        return output_features 
	
	    def forward(self, x , mask):
	        # x : 3 3 16 224 224 batch_size channel num_frames height weight
	        # mask : torch.Size([1, 16, 3, 3, 224, 224])
	        mask = mask.squeeze(0)  # shape: [16, 3, 3, 224, 224]
	        x = self.forward_features(x,mask)
	        x = self.head_dropout(x)
	        x = self.head(x)
	        # x.shape:3,2
	        return x

