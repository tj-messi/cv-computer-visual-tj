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

from transformers import GPT2Tokenizer, GPT2Model, ViTModel
from transformers import BlipTextModel, BlipConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PitVQANet(nn.Module):
    def __init__(self, num_class=18):  # 18/59
        super().__init__()

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
