#PitVQA

##代码git clone

直接下载然后上传即可

##数据集

在PitVQA_dataset目录里面

下载github的数据集

解压到如下目录

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1739281388289.png)

原数据集是video-25

先video-to-image

然后preprocess image处理好了内容

##流程跑通

记得几个load的pretrain模型，都下载下来然后换成本地路径

然后train

	python main.py --dataset=pit24 --epochs=60 --batch_size=64 --lr=0.00002

dataloader载入数据集

        # dataloader
        train_dataset = Pit24VQAClassification(train_seq, folder_head, folder_tail)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_dataset = Pit24VQAClassification(val_seq, folder_head, folder_tail)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

然后init初始化PitVQANet

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1739287317265.png)

其中视觉encoder

		# visual encoder
        model_name = "/data/coding/PitVQA-main/local_VIT"
        self.visual_encoder = ViTModel.from_pretrained(model_name)

文本encoder

        # text encoder
        config = BlipConfig.from_pretrained("/data/coding/PitVQA-main/local_Blip")
        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

tokenizer采用gpt2的tokenizer即可

       # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('/data/coding/PitVQA-main/local_gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # end of string

然后适应新的词汇表的长度：

先拿到gpt2tokenizer的长度，然后吧文本编码器的长度的部分嵌入到tokenizer的长度中，做了一个词汇表的匹配

		new_vocab_size = len(self.tokenizer)
		old_embeddings = self.text_encoder.embeddings.word_embeddings
		new_embeddings = nn.Embedding(new_vocab_size, old_embeddings.embedding_dim)
		new_embeddings.weight.data[:old_embeddings.num_embeddings, :] = old_embeddings.weight.data
		self.text_encoder.embeddings.word_embeddings = new_embeddings

然后用gpt2的decoder

	
然后过一个全连接层映射维度

		# intermediate layers
        self.intermediate_layer = nn.Linear(768, 512)
        self.se_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.2)

最后初始化一个分类头

	    # classifier
        self.classifier = nn.Linear(512, num_class)

初始化好了之后就开始train了
先拿到label，然后output需要进入到模型里面输入，并且输入question

	 for i, (_, images, questions, labels) in enumerate(train_dataloader, 0):
	        # labels
	        labels = labels.to(device)
	        outputs = model(image=images.to(device), question=questions)  # questions is a tuple
	        loss = criterion(outputs, labels)  # calculate loss
	        optimizer.zero_grad()
	        loss.backward()  # calculate gradient
	        optimizer.step()  # update parameters
	
	        # print statistics
	        total_loss += loss.item()
	
	        scores, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
	        if label_true is None:  # accumulate true labels of the entire training set
	            label_true = labels.data.cpu()
	        else:
	            label_true = torch.cat((label_true, labels.data.cpu()), 0)
	        if label_pred is None:  # accumulate pred labels of the entire training set
	            label_pred = predicted.data.cpu()
	        else:
	            label_pred = torch.cat((label_pred, predicted.data.cpu()), 0)
	        if label_score is None:
	            label_score = scores.data.cpu()
	        else:
	            label_score = torch.cat((label_score, scores.data.cpu()), 0)
	
	    # loss and acc
	    acc = calc_acc(label_true, label_pred)
	    precision, recall, f_score = calc_precision_recall_fscore(label_true, label_pred)
	    print(f'Train: epoch: {epoch} loss: {total_loss} | Acc: {acc} | '
	          f'Precision: {precision} | Recall: {recall} | F1 Score: {f_score}')
	    return acc

预测label的时候输入model执行forward

先输入图像和问题对

得到image的embedding之后再创建一个和image-embed一样维度但是全是1的向量

然后得到通过tokenizer输入的文本输入，将处理过的文本输入（text_inputs.input_ids 和 text_inputs.attention_mask）送入文本编码器（self.text_encoder）。
通过 encoder_hidden_states=image_embeds，文本编码器会同时接收图像的嵌入作为额外的信息（用于多模态学习）。
text_outputs.last_hidden_state 是文本编码器的输出，即文本输入的嵌入表示，存储在 text_output_embeds 中

将文本的嵌入（text_output_embeds）传递给 GPT-2 解码器（self.gpt_decoder）。
使用 encoder_attention_mask 作为注意力掩码，确保 GPT-2 解码器正确地处理输入的有效位置。
gpt_output.last_hidden_state 是解码器的输出，它包含了输入文本的生成或表示信息

首先将 decoder_output 张量的维度进行交换（swapaxes(1, 2)），然后进行自适应平均池化（adaptive_avg_pool1d），将每个序列的特征池化为一个固定大小的输出（维度为 1）。
最后，进行维度还原，并用 squeeze(1) 移除大小为 1 的维度

将池化后的 decoder_output 输入到中间层（self.intermediate_layer），这个层是一个全连接层，将输入从 768 维映射到 512 维。
然后使用 self.se_layer(out) 来应用一种叫做 "Squeeze-and-Excitation" 的机制，对输出进行逐元素乘法操作，可能是为了调整每个特征的权重。
接着进行批量归一化（LayerNorm），进一步稳定训练过程。
最后，应用 dropout（self.dropout(out)）来防止过拟合。

最后分类

	def forward(self, image, question):
	        image = image.to(device)
	
	        # visual encoder
	        image_embeds = self.visual_encoder(image).last_hidden_state
	        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
	
	        # get text features
	        text_inputs = self.tokenizer(question, return_tensors="pt", padding='max_length', max_length=30,
	                                     truncation=True).to(device)
	
	        # text encoder
	        text_outputs = self.text_encoder(
	            input_ids=text_inputs.input_ids,
	            attention_mask=text_inputs.attention_mask,
	            encoder_hidden_states=image_embeds,
	        )
	        text_output_embeds = text_outputs.last_hidden_state
	
	        # text decoder
	        gpt_output = self.gpt_decoder(inputs_embeds=text_output_embeds,
	                                      encoder_attention_mask=text_inputs.attention_mask)
	        decoder_output = gpt_output.last_hidden_state
	
	        # average pool
	        decoder_output = decoder_output.swapaxes(1, 2)
	        decoder_output = F.adaptive_avg_pool1d(decoder_output, 1)
	        decoder_output = decoder_output.swapaxes(1, 2).squeeze(1)
	
	        # intermediate layer
	        out = self.intermediate_layer(decoder_output)
	        out = torch.mul(out, self.se_layer(out))
	        out = self.LayerNorm(out)
	        out = self.dropout(out)
	
	        # classification layer
	        out = self.classifier(out)
	        return out


##创新点复现