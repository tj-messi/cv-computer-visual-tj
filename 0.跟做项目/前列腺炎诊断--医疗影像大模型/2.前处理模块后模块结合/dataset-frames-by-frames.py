#!/usr/bin/env python

import os, sys
from typing import Optional
# import av
import io
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from .transform import create_random_augment, random_resized_crop

class VideoDataset(torch.utils.data.Dataset):

    def __init__(
        self, list_path: str, data_root: str,
        num_spatial_views: int, num_temporal_views: int, random_sample: bool,
        num_frames: int, sampling_rate: int, spatial_size: int,
        mean: torch.Tensor, std: torch.Tensor,
        auto_augment: Optional[str] = None, interpolation: str = 'bicubic',
        mirror: bool = False,
    ):
        self.list_path = list_path
        self.data_root = data_root
        self.interpolation = interpolation
        self.spatial_size = spatial_size

        self.mean, self.std = mean, std
        self.num_frames, self.sampling_rate = num_frames, sampling_rate

        if random_sample:
            assert num_spatial_views == 1 and num_temporal_views == 1
            self.random_sample = True
            self.mirror = mirror
            self.auto_augment = auto_augment
        else:
            auto_augment = None
            mirror = False
            assert auto_augment is None and not mirror
            self.random_sample = False
            self.num_temporal_views = num_temporal_views 
            self.num_spatial_views = num_spatial_views

        with open(list_path) as f:
            # 只拿到了T0,1;Tx,x;的序列
            # ['NoCa/1,0', 'NoCa/2,0', 'NoCa/3,0', 'NoCa/4,0', 'NoCa/5,0', 'NoCa/6,0', 'NoCa/7,0',
            # 可以加上帧索引来添加内容
            self.data_list = f.read().splitlines()
            # 只有train训练时候需要拿连续帧
            if ('train'  in list_path) :
                data_list_with_index = []

                for line in self.data_list :
                    path, label = line.split(',')
                    root_now = self.data_root + '/' + path

                    files = os.listdir(root_now)
                    file_count = len([f for f in files if os.path.isfile(os.path.join(root_now, f))])
                    #print(file_count)

                    for i in range(file_count//8) :
                        data_list_with_index.append(line+','+ str((i)*num_frames))

                self.data_list = data_list_with_index
                # print(self.data_list)



    def __len__(self):
        return len(self.data_list)
    

    def __getitem__(self, idx):

        # 如果是训练集那么是T0/1----path   , 1-----label    ,  16-----开始帧标记
        line = self.data_list[idx]
        path=None
        label=None
        index=None
        if ('train' not in self.list_path):
            # print(line)
            index = ''
            path, label = line.split(',')
            ## 拿到的label
            path = os.path.join(self.data_root, path)
            # print(path)
            label = int(label)
        else:
             # print(line)
            path, label,index = line.split(',')
            ## 拿到的label
            path = os.path.join(self.data_root, path)
            # print(path)
            label = int(label)
            
        print(path,index,label)
        '''
        # 获取文件夹下所有帧
        frame_files = sorted(os.listdir(path))
        frames = {}
        for idx,filename in enumerate(frame_files):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                frame_path = os.path.join(path, filename)
                # with Image.open(frame_path) as img:
                img = Image.open(frame_path)
                pts = idx
                frames[pts] = img
        frames = [frames[k] for k in sorted(frames.keys())]
        # frames = [frames]
        # print(type(frames))
        # print(len(frames))# frames是一个list格式的数据
        '''
        # 尝试修改以获得更高的代码效率
        frame_files = sorted(os.listdir(path))
        # 直接加载所有图像到内存
        frames = []
        segment_path = []
        for filename in frame_files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                frame_path = os.path.join(path, filename)
                img = Image.open(frame_path)
                segment_path.append(frame_path)
                frames.append(img)
        
        new_test_folder = None
        
        '''
        container = av.open(path)
        frames = {}
        for frame in container.decode(video=0):
            frames[frame.pts] = frame
        container.close()
        frames = [frames[k] for k in sorted(frames.keys())]
        '''
        frames_list = []
        renewlabel_list = []
        if  self.random_sample:
            # frame_idx = self._sequential_sample_frame_idx(len(frames))
            frame_idx = list(range(int(index), int(index) + self.num_frames))
            renewlabel_list = frame_idx
            print(frame_idx,path)
            # frames = [frames[x].to_rgb().to_ndarray() for x in frame_idx]
            frames = np.array([np.array(frames[x].convert("RGB")) for x in frame_idx])
            # frames = [np.array(frames[x]) for x in frame_idx]
            # frames = np.array([np.array(frame.convert('RGB')) for frame in frames])
            frames = torch.as_tensor(np.stack(frames)).float() / 255.
            # print(path)

            if self.auto_augment is not None:
                aug_transform = create_random_augment(
                    input_size=(frames.size(1), frames.size(2)),
                    auto_augment=self.auto_augment,
                    interpolation=self.interpolation,
                )
                frames = frames.permute(0, 3, 1, 2) # T, C, H, W
                frames = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
                frames = aug_transform(frames)
                frames = torch.stack([transforms.ToTensor()(img) for img in frames])
                frames = frames.permute(0, 2, 3, 1)

            frames = (frames - self.mean) / self.std
            frames = frames.permute(3, 0, 1, 2) # C, T, H, W
            frames = random_resized_crop(
            frames, self.spatial_size, self.spatial_size,
            )
            frames_list.append(frames)
        else:  

            # frames = [x.to_rgb().to_ndarray() for x in frames]
            frames = np.array([np.array(frame.convert("RGB")) for frame in frames])
            frames = torch.as_tensor(np.stack(frames))
            frames = frames.float() / 255.

            frames = (frames - self.mean) / self.std
            frames = frames.permute(3, 0, 1, 2) # C, T, H, W
            
            if frames.size(-2) < frames.size(-1):
                new_width = frames.size(-1) * self.spatial_size // frames.size(-2)
                new_height = self.spatial_size
            else:
                new_height = frames.size(-2) * self.spatial_size // frames.size(-1)
                new_width = self.spatial_size
            frames = torch.nn.functional.interpolate(
                frames, size=(new_height, new_width),
                mode='bilinear', align_corners=False,
            )

            frames = self._generate_spatial_crops(frames)
            frames = sum([self._generate_temporal_crops(x) for x in frames], [])
            if len(frames) > 1:
                frames = torch.stack(frames)
            

        # print(len(frames))
        # labels = torch.full((frames.size(0),), label, dtype=torch.long)
        # print(labels.shape)

        ''' if label == 0 :
            frames_segment = frames
            # 所有frames存到medsam2文件中按照test格式排版
            base_dir = '/root/prostate-check-zjz/Medical-SAM2-zjz/Test-Medsam2'
            if not os.path.exists(base_dir):
                # 如果不存在，则创建文件夹
                os.makedirs(base_dir)
            num_existing_folders = len([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
            i = num_existing_folders + 1

            new_test_folder = os.path.join(base_dir, f'Test-{i}')
            os.makedirs(new_test_folder, exist_ok=True)      
            test_subfolder = os.path.join(new_test_folder, 'Test')
            os.makedirs(test_subfolder, exist_ok=True)

            for idx,frame in enumerate(frames_segment) :
                frame_folder = os.path.join(test_subfolder, f'idx_{idx+1}') 
                os.makedirs(frame_folder, exist_ok=True)
                            
                img_path = os.path.join(frame_folder, f'idx_{idx+1}_img.png')
                label_path = os.path.join(frame_folder, f'idx_{idx+1}_label.png')

                (frame).save(img_path)
                (frame).save(label_path)            
        #zjz-2025-1-24'''

        renewlabel_list.append(path)
        renewlabel_list.append(new_test_folder)
        #print(renewlabel_list)

        return frames, label
        
        '''labels = torch.full(frames.shape[0],label,dtype=torch.long)
        return frames,labels'''


    def _generate_temporal_crops(self, frames):
        seg_len = (self.num_frames - 1) * self.sampling_rate + 1
        if frames.size(1) < seg_len:
            frames = torch.cat([frames, frames[:, -1:].repeat(1, seg_len - frames.size(1), 1, 1)], dim=1)
        slide_len = frames.size(1) - seg_len

        crops = []
        for i in range(self.num_temporal_views):
            if self.num_temporal_views == 1:
                st = slide_len // 2
            else:
                st = round(slide_len / (self.num_temporal_views - 1) * i)

            crops.append(frames[:, st: st + self.num_frames * self.sampling_rate: self.sampling_rate])
        
        return crops


    def _sequential_sample_frame_idx(self, len):
        frame_indices = []

        if self.sampling_rate < 0:  # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(start)  # 顺序采样，不随机选择
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            # 当采样间隔太大，无法保证每一帧的顺序采样时，使用一种修正策略
            for i in range(self.num_frames):
                # 顺序采样，确保每一帧都按顺序选取
                frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            # 顺序选择采样位置
            # 选择起始位置，从0开始，每次按顺序采样
            start = 0  # 顺序采样，从第一个开始
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices

    def _generate_spatial_crops(self, frames):
        if self.num_spatial_views == 1:
            assert min(frames.size(-2), frames.size(-1)) >= self.spatial_size
            h_st = (frames.size(-2) - self.spatial_size) // 2
            w_st = (frames.size(-1) - self.spatial_size) // 2
            h_ed, w_ed = h_st + self.spatial_size, w_st + self.spatial_size
            return [frames[:, :, h_st: h_ed, w_st: w_ed]]

        elif self.num_spatial_views == 3:
            assert min(frames.size(-2), frames.size(-1)) == self.spatial_size
            crops = []
            margin = max(frames.size(-2), frames.size(-1)) - self.spatial_size
            for st in (0, margin // 2, margin):
                ed = st + self.spatial_size
                if frames.size(-2) > frames.size(-1):
                    crops.append(frames[:, :, st: ed, :])
                else:
                    crops.append(frames[:, :, :, st: ed])
            return crops
        
        else:
            raise NotImplementedError()


    def _random_sample_frame_idx(self, len):
        frame_indices = []

        if self.sampling_rate < 0: # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(np.random.randint(start, end + 1))
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            for i in range(self.num_frames):
                frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            start = np.random.randint(len - self.sampling_rate * (self.num_frames - 1))
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, list_path: str, num_frames: int, num_views: int, spatial_size: int):
        with open(list_path) as f:
            self.len = len(f.read().splitlines())
        self.num_frames = num_frames
        self.num_views = num_views
        self.spatial_size = spatial_size

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        shape = [3, self.num_frames, self.spatial_size, self.spatial_size]
        if self.num_views != 1:
            shape = [self.num_views] + shape
        return torch.zeros(shape), 0
