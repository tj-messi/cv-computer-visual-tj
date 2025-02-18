'''
Description: Model implementation of PitVQA-Net model
Paper: PitVQA: Image-grounded Text Embedding LLM for Visual Question Answering in Pituitary Surgery
Author: Runlong He, Mengya Xu, Adrito Das, Danyal Z. Khan, Sophia Bano, 
        Hani J. Marcus, Danail Stoyanov, Matthew J. Clarkson, Mobarakol Islam
Lab: Wellcome/EPSRC Centre for Interventional and Surgical Sciences (WEISS), UCL
Acknowledgement : Code adopted from the official implementation of 
                  Huggingface Transformers (https://github.com/huggingface/transformers)
                  and Surgical-GPT (https://github.com/lalithjets/SurgicalGPT).
'''

import torch
from torch import nn
import torch.nn.functional as F

from transformers import GPT2Tokenizer, GPT2Model, ViTModel, VivitModel, VivitConfig
from transformers import BlipTextModel, BlipConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VideoFeatureExtractor(nn.Module):
    def __init__(self):
        super(VideoFeatureExtractor, self).__init__()
        configuration = VivitConfig()
        self.vivit_model = VivitModel.from_pretrained("/home/test/PitVQA-main/local_VIVIT")
        self.vivit_model.num_frames = 37

    def forward(self, video_frames):
        # video_frames : torch.Size([16, 37, 3, 224, 224])
        outputs = self.vivit_model(video_frames)
        # Extract the last hidden state (pooled features)
        video_features = outputs.last_hidden_state.mean(dim=1)  # Average over frames
        return video_features


class PitVQANet(nn.Module):
    def __init__(self, num_class=18):  # 18/59
        super().__init__()

        # Video feature extraction module
        self.video_feature_extractor = VideoFeatureExtractor()
        
        # visual encoder
        model_name = "/home/test/PitVQA-main/local_VIT"
        self.visual_encoder = ViTModel.from_pretrained(model_name)

        # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('/home/test/PitVQA-main/local_gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # end of string

        # text encoder
        config = BlipConfig.from_pretrained("/home/test/PitVQA-main/local_Blip")
        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        new_vocab_size = len(self.tokenizer)
        old_embeddings = self.text_encoder.embeddings.word_embeddings
        new_embeddings = nn.Embedding(new_vocab_size, old_embeddings.embedding_dim)
        new_embeddings.weight.data[:old_embeddings.num_embeddings, :] = old_embeddings.weight.data
        self.text_encoder.embeddings.word_embeddings = new_embeddings

        # decoder
        self.gpt_decoder = GPT2Model.from_pretrained('/home/test/PitVQA-main/local_gpt2')

        # intermediate layers
        self.intermediate_layer = nn.Linear(768, 512)
        self.se_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.2)

        # classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self, image, video, question):
        image = image.to(device)
        video = video.to(device)
        # Video feature extraction
        # videos = torch.Size([37, 16, 3, 224, 224])
        video = video.permute(1, 0, 2, 3, 4) # ViviT 接受 (batch_size,num_frames,channel,width,hight)
        video_embeds = self.video_feature_extractor(video) 

        # visual encoder 
        image_embeds = self.visual_encoder(image).last_hidden_state # torch.Size([64, 197, 768])
        # image_embeds: torch.Size([4, 197, 768])
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) # torch.Size([64, 197])

        # get text features
        text_inputs = self.tokenizer(question, return_tensors="pt", padding='max_length', max_length=30,
                                     truncation=True).to(device)

        # combine image-video
        # video_embeds : torch.Size([4, 768])
        # combined_embeds :  torch.Size([4, 198, 768])
        video_embeds = video_embeds.unsqueeze(1)  # Shape: [batch_size, 1, hidden_size]
        combined_embeds = torch.cat((image_embeds, video_embeds), dim=1)

        # text encoder 
        # combined_embeds :  torch.Size([4, 198, 768])
        text_outputs = self.text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            encoder_hidden_states=combined_embeds,
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
