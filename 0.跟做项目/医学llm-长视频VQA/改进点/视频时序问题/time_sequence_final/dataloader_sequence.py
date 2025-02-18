'''
Description: Dataloader of PitVQA-Net model
Paper: PitVQA: Image-grounded Text Embedding LLM for Visual Question Answering in Pituitary Surgery
Author: Runlong He, Mengya Xu, Adrito Das, Danyal Z. Khan, Sophia Bano, 
        Hani J. Marcus, Danail Stoyanov, Matthew J. Clarkson, Mobarakol Islam
Lab: Wellcome/EPSRC Centre for Interventional and Surgical Sciences (WEISS), UCL
Acknowledgement : Code adopted from the official implementation of 
                  Huggingface Transformers (https://github.com/huggingface/transformers)
                  and Surgical-GPT (https://github.com/lalithjets/SurgicalGPT).
'''

import os
import glob

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path
from torchvision.transforms.functional import InterpolationMode
import cv2

from utils_sequence import *

class EndoVis18VQAGPTClassification(Dataset):
    def __init__(self, seq, folder_head, folder_tail, transform=None):
        # define transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # get files, questions and answers
        filenames = []
        for curr_seq in seq:
            filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines:
                self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' % (len(filenames), len(self.vqas)))

        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                       'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction',
                       'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                       'left-top', 'right-top', 'left-bottom', 'right-bottom']

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        # get path
        qa_full_path = Path(self.vqas[idx][0])
        seq_path = qa_full_path.parents[2]
        file_name = self.vqas[idx][0].split('/')[-1]

        # img
        img_loc = os.path.join(seq_path, 'left_fr', file_name.split('_')[0] + '.png')
        raw_image = Image.open(img_loc).convert('RGB')
        img = self.transform(raw_image)

        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        answer = self.vqas[idx][1].split('|')[1]
        label = self.labels.index(str(answer))

        return img_loc, img, question, label


class Pit24VQAClassification(Dataset):
    def __init__(self, seq, folder_head, folder_tail):

        self.transform = transforms.Compose([
            # transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

        # files, question and answers
        filenames = []
        for curr_seq in seq:
            filenames = filenames + glob.glob(folder_head + curr_seq + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines:
                answer = line.split('|')[1]
                if answer not in ['no_visible_instrument', 'no_secondary_instrument']:  # filter unknown answers
                    self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' % (len(filenames), len(self.vqas)))

        # Labels
        self.labels = [
            #0-14
            'nasal_corridor_creation', 'anterior_sphenoidotomy', 'septum_displacement', 'sphenoid_sinus_clearance',
            'sellotomy', 'haemostasis', 'synthetic_graft_placement', 'durotomy', 'tumour_excision',
            'fat_graft_placement', 'gasket_seal_construct', 'dural_sealant', 'nasal_packing', 'debris_clearance',
            'end_of_step',  # 15 steps
            #15-18
            'nasal_sphenoid', 'sellar', 'closure',  'end_of_phase',  # 4 phases
            #19-36
            'suction', 'freer_elevator', 'pituitary_rongeurs', 'spatula_dissector', 'kerrisons', 'cottle',
            'haemostatic_foam', 'micro_doppler', 'nasal_cutting_forceps', 'stealth_pointer', 'irrigation_syringe',
            'retractable_knife', 'dural_scissors', 'ring_curette', 'cup_forceps', 'bipolar_forceps', 'tissue_glue',
            'surgical_drill',  # 18 instruments
            #37-39
            'zero', 'one', 'two',  # 3 number of instruments
            #40-44
            'top-left', 'top-right', 'centre', 'bottom-left', 'bottom-right',  # 5 positions
            #45-58
            'The middle and superior turbinates are laterally displaced',
            'The sphenoid ostium is identified and opened', 'The septum is displaced until the opposite ostium is seen',
            'The sphenoid sinus is opened, with removal of sphenoid septations to expose the face of the sella and mucosa',
            'Haemostasis is achieved with a surgiflo, a bipolar cautery, and a spongostan placement',
            'The sella is identified, confirmed and carefully opened', 'A cruciate durotomy is performed',
            'The tumour is seen and removed in a piecemeal fashion', 'spongostan, tachosil and duragen placement',
            'A fat graft is placed over the defact', 'Evicel and Adherus dural sealant are applied',
            'Debris is cleared from the nasal cavity and choana', 'A MedPor implant and a fascia lata graft are placed',
            'The nasal cavity is packed with Bismuth soaked ribbon gauze'  # 14 operations

        ]

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        qa_full_path = Path(self.vqas[idx][0])
        seq_path = qa_full_path.parents[2]
        video_num = qa_full_path.parts[-2]
        file_name = self.vqas[idx][0].split('/')[-1]
        # vqas : ['/home/test/PitVQA-main/PitVQA_dataset/qa-classification/video_01/01514.txt', 
        # 'What is the surgical operation performed in the image?|
        # The sphenoid sinus is opened, with removal of sphenoid septations to expose the face of the sella and mucosa']

        # img torch.Size([64, 3, 224, 224])
        img_loc = os.path.join(seq_path, 'images', video_num, file_name.split('.')[0] + '.png')
        # /home/test/PitVQA-main/PitVQA_dataset + '/' + 'images' + '/' + 'video_0x' + '/' + 'xxxxx' + '.png'
        raw_image = Image.open(img_loc).convert('RGB')
        img = self.transform(raw_image)

        # question and answer
        # How many instruments are present in the image?|one
        
        question = self.vqas[idx][1].split('|')[0]
        answer = self.vqas[idx][1].split('|')[1]
        label = self.labels.index(str(answer))

        # video [num_frames, channels, height, width]
        raw_video = []
        raw_video.append(raw_image)
        selected_txt = os.path.join(seq_path, 'qa-classification', video_num, file_name.split('.')[0] + '.txt')
        selected_idx = int(file_name.split('.')[0]) # xxxxx 转化为整数
        step = 1
        while (selected_idx - step >= 0) :
            left_idx = (selected_idx - step)
            left_idx_format_num = f"{left_idx:05d}"
            left_txt_path = os.path.join(seq_path , 'qa-classification' , video_num,left_idx_format_num + '.txt')
            if not os.path.exists(left_txt_path) :
                break
            if compare_files(selected_txt,left_txt_path) :
                left_frame_path = os.path.join(seq_path, 'images', video_num, left_idx_format_num + '.png')
                left_frame = Image.open(left_frame_path).convert('RGB')
                raw_video.insert(0,left_frame)
            else:
                break
            step+=1
        step = 1
        while (1) :
            right_idx = (selected_idx + step)
            right_idx_format_num = f"{right_idx:05d}"
            right_txt_path = os.path.join(seq_path , 'qa-classification' , video_num,right_idx_format_num + '.txt')
            if not os.path.exists(right_txt_path) :
                break
            if compare_files(selected_txt,right_txt_path) :
                right_frame_path = os.path.join(seq_path, 'images', video_num, right_idx_format_num + '.png')
                right_frame = Image.open(right_frame_path).convert('RGB')
                raw_video.append(right_frame)
            else:
                break
            step+=1
        # 最后根据95分位进行裁剪和填充
        raw_video = padding_and_slicing(raw_video,4)
        # print("img_location: ",img_loc," video-length: ",len(raw_video))
        transformed_video = [self.transform(frame) for frame in raw_video]
        '''
        # 2025-2-12 zjz
        # question and answer (CoT improved)
        How many instruments are present in the image?|i can only see one freer_elevator in the centre of the image so the answer is one|one

        question = self.vqas[idx][1].split('|')[0]
        answer = self.vqas[idx][1].split('|')[1]
        label_str = self.vqas[idx][1].split('|')[2]
        label = self.labels.index(str(label_str))
        '''

        return img_loc, img, transformed_video, question, label
