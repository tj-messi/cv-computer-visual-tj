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


#


#clip数量问题

可以按照论文里面这样说的：

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1743734586518.png)

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1743734600723.png)