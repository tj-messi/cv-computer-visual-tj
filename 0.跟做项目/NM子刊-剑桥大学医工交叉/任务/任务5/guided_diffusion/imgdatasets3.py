import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
from skimage.measure import shannon_entropy
from PIL import Image,ImageOps
import scipy.sparse as sp_sparse
def np_norm(inp):
    max_in=np.max(inp)
    min_in = np.min(inp)
    return (inp-min_in)/(max_in-min_in)

def gray_value_of_gene(gene_class,gene_order):
    gene_order=list(gene_order)
    Index=gene_order.index(gene_class)
    interval=255/len(gene_order)
    value=Index*interval
    return int(value)

import numpy as np



import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
from skimage.measure import shannon_entropy
from PIL import Image,ImageOps

def load_data(data_root,dataset_use,status,SR_times,gene_num,all_gene,gene_order=None,gene_name_order=None
):

    if dataset_use=='Xenium':
        dataset= Xenium_dataset(data_root,SR_times,status,gene_num,all_gene)
    elif dataset_use=='SGE':
        dataset= SGE_dataset(data_root,gene_num,all_gene)
    elif dataset_use=='BreastST':
        dataset= BreastST_dataset(data_root,gene_num,all_gene)
    elif dataset_use =='Xenium5k':
       dataset= Xenium5k_dataset(data_root,dataset_use,SR_times,status,gene_num,all_gene,gene_order,gene_name_order)
    return dataset

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

# 假设已有 gray_value_of_gene 函数

import os
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
from skimage.measure import shannon_entropy
from PIL import Image, ImageOps
import scipy.sparse as sp_sparse
from transformers import AutoTokenizer, AutoModel

def np_norm(inp):
    max_in = np.max(inp)
    min_in = np.min(inp)
    return (inp - min_in) / (max_in - min_in)

def gray_value_of_gene(gene_class, gene_order):
    gene_order = list(gene_order)
    Index = gene_order.index(gene_class)
    interval = 255 / len(gene_order)
    value = Index * interval
    return int(value)

class Xenium5k_dataset(Dataset):
    def __init__(self, data_root,dataset_use, SR_times, status, gene_num, all_gene, gene_order, gene_name_order):
        """
        data_root: 数据根目录路径
        SR_times: 下采样倍数（例如10或5）
        status: 数据集状态，'Train' 或 'Test'
        gene_num: 每个样本需要处理的基因数量
        all_gene: 总基因数（用于reshape等）
        gene_order: 基因排序索引（用于通道选择）
        gene_name_order: 基因名称列表或包含名称的txt文件路径
        """
        # 固定使用一个样本ID，并按 patch 划分训练与测试
        sample_name = ['20240611mouseembryo']
        self.selected_patches = []  # 存储符合条件的 (sample_id, patch_id)
        self.gene_order = gene_order
        self.gene_name_order = gene_name_order
        self.gene_num = gene_num

        # 根据 patch 名称中下划线分割后的第二部分判断 patch 编号
        for sample_id in sample_name:
            # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
            base_path = f"{data_root}/HR_ST/extract/{sample_id}/"
            print('base_path',base_path)
            sub_patches = os.listdir(base_path)
            # 第一步：收集该样本的所有有效行号
            row_numbers = []
            for patch_id in sub_patches:
                parts = patch_id.split('_')
                if len(parts) < 2:
                    continue  # 跳过非法格式
                try:
                    b = int(parts[1])  # 提取行号
                    row_numbers.append(b)
                except ValueError:
                    continue  # 跳过无法转换为整数的情况
            
            # 第二步：计算分界点（获取独特的行号并排序）
            unique_rows = sorted(set(row_numbers))
            total_rows = len(unique_rows)
            # print(total_rows,total_rows)
            # 第三步：根据总行数确定训练/测试分界点
            if total_rows <= 5:
                # 如果总行数≤4，最后1行用9于测试
                train_max_row = unique_rows[-2] if total_rows > 1 else -1
            elif total_rows <= 10:
                # 如果总行数在5-8之间，最后2行用于测试
                train_max_row = unique_rows[-3]
            else:
                # 如果行数更多，大约20%用于测试（向上取整）
                test_count = max(int(total_rows * 0.2 + 0.5), 3)  # 至少3行测试
                train_max_row = unique_rows[-(test_count + 1)]
            # print(train_max_row)
            # 第四步：遍历patch并应用新的划分逻辑
            for patch_id in sub_patches:
                parts = patch_id.split('_')
                if len(parts) < 2:
                    continue  # 跳过非法格式
                
                try:
                    b = int(parts[1])  # 提取行号
                    
                    # 应用新的动态划分规则
                    # if status == 'Train' and (train_max_row == -1 or b <= train_max_row):
                    #     # print('patch_id',patch_id)
                    #     self.selected_patches.append((sample_id, patch_id))
                    # elif status == 'Test' and (train_max_row == -1 or b > train_max_row):
                    #     self.selected_patches.append((sample_id, patch_id))
                    
                    # test取全体
                    if status == 'Train' and (train_max_row == -1 or b <= train_max_row):
                        # print('patch_id',patch_id)
                        self.selected_patches.append((sample_id, patch_id))
                    elif status == 'Test':
                        self.selected_patches.append((sample_id, patch_id))

                except ValueError:
                    continue  # 跳过无法转换为整数的情况
                    
        # 加载 HR ST 数据
        SR_ST_all = []
        self.gene_scale = []
        for sample_id, patch_id in self.selected_patches:
            hr_st_dir = os.path.join(data_root,  "HR_ST", "extract", sample_id, patch_id)
            if SR_times == 10:
                SR_ST = sp_sparse.load_npz(os.path.join(data_root,'HR_ST/extract', sample_id, patch_id, 'HR_ST_256.npz')).toarray().reshape(256, 256, 10417)
                SR_ST = np.transpose(SR_ST, axes=(2, 0, 1))  # 转为 (channels, H, W)
            elif SR_times == 5:
                SR_ST = sp_sparse.load_npz(os.path.join(data_root,'HR_ST/extract', sample_id, patch_id, 'HR_ST_128.npz')).toarray().reshape(128, 128, 10417)
                SR_ST = np.transpose(SR_ST, axes=(2, 0, 1))
            SR_ST_all.append(SR_ST)
            self.gene_scale.append(gene_order)
        self.SR_ST_all = np.array(SR_ST_all).astype(np.float64)
        self.SR_ST_all = self.SR_ST_all[:, gene_order, ...]
        self.SR_ST_all_groups = self.SR_ST_all.copy()
        # 对 HR ST 数据归一化（逐 patch、逐通道）
        for ii in range(self.SR_ST_all_groups.shape[0]):
            for jj in range(self.SR_ST_all_groups.shape[1]):
                if np.sum(self.SR_ST_all_groups[ii, jj]) != 0:
                    Max = np.max(self.SR_ST_all_groups[ii, jj])
                    Min = np.min(self.SR_ST_all_groups[ii, jj])
                    self.SR_ST_all_groups[ii, jj] = (self.SR_ST_all_groups[ii, jj] - Min) / (Max - Min + 1e-8)
                    
        # 加载 spot ST 数据
        spot_ST_all = []
        for sample_id, patch_id in self.selected_patches:
            spot_ST = sp_sparse.load_npz(os.path.join(data_root, 'spot_ST/extract', sample_id, patch_id, 'spot_ST.npz')).toarray().reshape(26, 26, 10417)
            spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
            spot_ST_all.append(spot_ST)
        self.spot_ST_all = np.array(spot_ST_all).astype(np.float64)
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...]
        self.spot_ST_all_groups = self.spot_ST_all.copy()
        

        for ii in range(self.spot_ST_all_groups.shape[0]):
            for jj in range(self.spot_ST_all_groups.shape[1]):
                if np.sum(self.spot_ST_all_groups[ii, jj]) != 0:
                    Max = np.max(self.spot_ST_all_groups[ii, jj])
                    Min = np.min(self.spot_ST_all_groups[ii, jj])
                    self.spot_ST_all_groups[ii, jj] = (self.spot_ST_all_groups[ii, jj] - Min) / (Max - Min + 1e-8)
                    
        # 加载 WSI 5120 数据
        WSI_5120_all = []
        for sample_id, patch_id in self.selected_patches:
            wsi_path = os.path.join(data_root, "WSI", "extract", sample_id, patch_id, "5120_to256.npy")
            WSI_5120 = np.load(wsi_path)
            WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
            WSI_5120_all.append(WSI_5120)
        self.WSI_5120_all = np.array(WSI_5120_all)
        self.WSI_5120_all_expanded = self.WSI_5120_all.copy()
        
        # 加载 WSI 320 数据
        WSI_320_all = []
        for sample_id, patch_id in self.selected_patches:
            wsi_320_path = os.path.join(data_root,"WSI", "extract", sample_id, patch_id, "320_to16.npy")
            WSI_320 = np.load(wsi_320_path)
            WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
            WSI_320_all.append(WSI_320)
        self.WSI_320_all = np.array(WSI_320_all)
        self.WSI_320_all_expanded = self.WSI_320_all.copy()
        
        # 加载 BERT 模型，提取基因名称特征
        self.tokenizer = AutoTokenizer.from_pretrained("/media/cbtil/T7 Shield/NMI/bert",
                                                         local_files_only=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained("/media/cbtil/T7 Shield/NMI/bert",
                                                 local_files_only=True, trust_remote_code=True)
        self.model.eval()
        
        if isinstance(gene_name_order, str) and os.path.exists(gene_name_order):
            with open(gene_name_order, "r") as f:
                gene_names = [f"This gene’s name is called {line.strip()}." for line in f if line.strip()]
        else:
            gene_names = [f"This gene’s name is called {gene}." for gene in gene_name_order]
        print("前5个基因名称句子：", gene_names[:5])
        
        gene_feats = []
        with torch.no_grad():
            for gene in gene_names:
                inputs = self.tokenizer(gene, return_tensors="pt")
                outputs = self.model(**inputs)
                gene_embedding = outputs.pooler_output.squeeze(0)
                gene_feats.append(gene_embedding)
        self.gene_name_features = torch.stack(gene_feats, dim=0)
        print('gene_name_features shape:', self.gene_name_features.shape)
        
        metadata_prompt = ("Provide spatial transcriptomics data from the VisiumHD platform for mouse species, with a healthy condition, and Whole Embryo tissue type.")
        with torch.no_grad():
            inputs = self.tokenizer(metadata_prompt, return_tensors="pt")
            outputs = self.model(**inputs)
            self.metadata_feature = outputs.pooler_output.squeeze(0)
        print('metadata_feature shape:', self.metadata_feature.shape)

    def __len__(self):
        return len(self.selected_patches)

    def __getitem__(self, index):
        # 生成基因位置编码（每个基因对应一张 256x256 的灰度图）
        gene_class = self.gene_scale[index]
        Gene_index_maps = [
            np.ones((256, 256, 1)) * gray_value_of_gene(code, self.gene_order) / 255.0
            for code in gene_class
        ]
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2).transpose(2, 0, 1)
        # 返回 HR ST、spot ST、WSI 5120、WSI 320、基因位置编码、基因名称特征和元信息特征
        return (
            self.SR_ST_all[index],
            self.spot_ST_all[index],
            self.WSI_5120_all[index],
            self.WSI_320_all[index],
            gene_class,
            final_Gene_index_map,
            self.gene_name_features,   # shape: [gene_num, hidden_size]
            self.metadata_feature       # shape: [hidden_size]
        )

class Xenium_dataset(Dataset):
    def __init__(self, data_root, SR_times, status, gene_num,all_gene):
        '''
            data_root: 数据根目录的路径。
            SR_times: 下采样倍数，影响加载的HR ST数据的分辨率。
            status: 指定数据集的状态，值为 'Train' 或 'Test'，用于选择不同的样本。
            gene_num: 需要处理的基因数量。
        '''
        if status == 'Train':
            sample_name = ['01220101', '01220102', 'NC1', 'NC2']#, '0418'

        elif status == 'Test':
            sample_name = ['01220201', '01220202']

        self.all_gene = all_gene
        gene_order_path = os.path.join(data_root, 'gene_order1.npy')
        gene_order = np.load(gene_order_path)[0:all_gene]
        # print(gene_order)
        self.gene_order = gene_order
        SR_ST_all = []
        self.gene_scale = []#new1.5:
        self.gene_num = gene_num#new1.5:
        
        # 
        # 
        ### HR ST
        for sample_id in sample_name:
            sub_patches=os.listdir(data_root+'Xenium/HR_ST/extract/'+sample_id)
            for patch_id in sub_patches:
                if SR_times==10:
                    print(data_root+'Xenium/HR_ST/extract/'+sample_id+'/'+patch_id+'/HR_ST_256.npz')
                    SR_ST=sp_sparse.load_npz(data_root+'Xenium/HR_ST/extract/'+sample_id+'/'+patch_id+'/HR_ST_256.npz').toarray().reshape(256, 256, 280)
                elif SR_times==5:
                    SR_ST = sp_sparse.load_npz(data_root+'Xenium/HR_ST/extract/'+sample_id+'/'+patch_id+'/HR_ST_128.npz').toarray().reshape(128, 128, 280)
                SR_ST=np.transpose(SR_ST,axes=(2,0,1))
                SR_ST_all.append(SR_ST)
                self.gene_scale.append(gene_order)#new1.5:
        self.SR_ST_all = np.array(SR_ST_all)
        #new1.5:
        self.gene_scale = np.array(self.gene_scale)
        # self.gene_scale = np.reshape(self.gene_scale, (self.gene_scale.shape[0]*(all_gene//gene_num),gene_num))
        #new1.5:
        # 新
        self.SR_ST_all = self.SR_ST_all[:, gene_order, ...].astype(np.float64)

        # print('SR_ST_all',self.SR_ST_all.shape)
        # self.SR_ST_all = np.reshape(self.SR_ST_all, (self.SR_ST_all.shape[0]*(all_gene//gene_num),gene_num, self.SR_ST_all.shape[2],self.SR_ST_all.shape[3]))
        # print('SR_ST_all',self.SR_ST_all.shape)
        for ii in range(self.SR_ST_all.shape[0]):  # 初始化SR_ST_all列表，用于存储加载的HR ST数据。
            for jj in range(self.SR_ST_all.shape[1]):
                if np.sum(self.SR_ST_all[ii, jj])!= 0:
                    Max = np.max(self.SR_ST_all[ii, jj])
                    Min = np.min(self.SR_ST_all[ii, jj])
                    self.SR_ST_all[ii, jj] = (self.SR_ST_all[ii, jj] - Min) / (Max - Min)  # 对选择的基因进行归一化处理

        ### spot ST
        spot_ST_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Xenium/spot_ST/extract/' + sample_id)
            for patch_id in sub_patches:
                spot_ST = np.load(data_root + 'Xenium/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
                spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
                spot_ST_all.append(spot_ST)
        self.spot_ST_all = np.array(spot_ST_all)

        # 新
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)

        # print('spot_ST_all',self.spot_ST_all.shape)
        # self.spot_ST_all = np.reshape(self.spot_ST_all, (self.spot_ST_all.shape[0]*(all_gene//gene_num),gene_num, self.spot_ST_all.shape[2],self.spot_ST_all.shape[3]))
        # print('spot_ST_all',self.spot_ST_all.shape)
        for ii in range(self.spot_ST_all.shape[0]):
            for jj in range(self.spot_ST_all.shape[1]):
                if np.sum(self.spot_ST_all[ii, jj])!= 0:
                    Max = np.max(self.spot_ST_all[ii, jj])
                    Min = np.min(self.spot_ST_all[ii, jj])
                    self.spot_ST_all[ii, jj] = (self.spot_ST_all[ii, jj] - Min) / (Max - Min)

        ### WSI 5120
        WSI_5120_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Xenium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_5120 = np.load(data_root + 'Xenium/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
                max = np.max(WSI_5120)
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
        self.WSI_5120_all = np.array(WSI_5120_all)
             # 对WSI 5120数据按照分组数量复制到batch维度
        # self.WSI_5120_all = np.repeat(self.WSI_5120_all, all_gene//gene_num, axis = 0)

    def __len__(self):
        # 返回处理后数据的batch维度大小（也就是分组后的数量）
        return self.SR_ST_all.shape[0]

    def __getitem__(self, index):
        '''
            返回对应索引位置处理后的各项数据，包含分组融入batch维度后的ST数据以及对应复制后的WSI数据。
        ''' 
        #new1.5:
        gene_class = self.gene_scale[index]

        Gene_index_maps = []

        for gene_code in gene_class:
            Gene_codes = gray_value_of_gene(gene_code, self.gene_order)
            Gene_index_map = np.ones(shape=(256, 256,1)) * Gene_codes / 255.0
            Gene_index_maps.append(Gene_index_map)
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2)
        final_Gene_index_map = np.moveaxis(final_Gene_index_map, 2, 0)
        #new1.5:

        return self.SR_ST_all[index], self.spot_ST_all[index], self.WSI_5120_all[index], final_Gene_index_map #new1.5:, self.WSI_320_all[index],



class SGE_dataset(Dataset):
    def __init__(self, data_root, gene_num):
        sample_name = ['0701', '0106']

        # 加载全部基因顺序信息，不再截取前gene_num个
        all_gene=30
        gene_order_path = os.path.join(data_root, 'gene_order.npy')
        gene_order = np.load(gene_order_path)[0:all_gene]
        SR_ST_all = []

        ### spot ST
        spot_ST_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Visium/spot_ST/extract/' + sample_id)
            for patch_id in sub_patches:
                spot_ST = np.load(data_root + 'Visium/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
                spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
                spot_ST_all.append(spot_ST)
        self.spot_ST_all = np.array(spot_ST_all)

        # 新
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)

        # print('spot_ST_all',self.spot_ST_all.shape)
        self.spot_ST_all = np.reshape(self.spot_ST_all, (self.spot_ST_all.shape[0]*(all_gene//gene_num),gene_num, self.spot_ST_all.shape[2],self.spot_ST_all.shape[3]))
        # print('spot_ST_all',self.spot_ST_all.shape)
        for ii in range(self.spot_ST_all.shape[0]):
            for jj in range(self.spot_ST_all.shape[1]):
                if np.sum(self.spot_ST_all[ii, jj])!= 0:
                    Max = np.max(self.spot_ST_all[ii, jj])
                    Min = np.min(self.spot_ST_all[ii, jj])
                    self.spot_ST_all[ii, jj] = (self.spot_ST_all[ii, jj] - Min) / (Max - Min)

        ### WSI 5120
        WSI_5120_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Visium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_5120 = np.load(data_root + 'Visium/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
                max = np.max(WSI_5120)
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
        self.WSI_5120_all = np.array(WSI_5120_all)
             # 对WSI 5120数据按照分组数量复制到batch维度
        self.WSI_5120_all = []
        for _ in range(all_gene//gene_num):
            self.WSI_5120_all.append(self.WSI_5120_all)
        self.WSI_5120_all = np.concatenate(self.WSI_5120_all, axis=0)

        ### WSI 320
        self.num_320 = []
        WSI_320_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Visium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_320 = np.load(data_root + 'Visium/WSI/extract/' + sample_id + '/' + patch_id + '/320_to16.npy')
                WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
                WSI_320_all.append(WSI_320)
        self.WSI_320_all = np.array(WSI_320_all)

        # 对WSI 320数据按照分组数量复制到batch维度
        self.WSI_320_all = []
        for _ in range(all_gene//gene_num):
            self.WSI_320_all.append(self.WSI_320_all)
        self.WSI_320_all = np.concatenate(self.WSI_320_all, axis=0)
        max_320 = np.max(WSI_320)
        a = 1
    def __len__(self):
        # 返回处理后数据的batch维度大小（也就是分组后的数量）
        return self.spot_ST_all.shape[0]

    def __getitem__(self, index):
        '''
            返回对应索引位置处理后的各项数据，包含分组融入batch维度后的ST数据以及对应复制后的WSI数据。
        '''
        return self.spot_ST_all[index], self.WSI_5120_all[index], self.WSI_320_all[index]


class BreastST_dataset(Dataset):
    def __init__(self, data_root, gene_num):
        from skimage.transform import resize
        sample_name = os.listdir(data_root + 'NBME/spot_ST/extract/')

        all_gene=130
        gene_order_path = os.path.join(data_root, 'gene_order.npy')
        gene_order = np.load(gene_order_path)[0:all_gene]
        SR_ST_all = []

        ### spot ST
        spot_ST_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'NBME/spot_ST/extract/' + sample_id)
            for patch_id in sub_patches:
                spot_ST = np.load(data_root + 'NBME/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
                spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
                spot_ST_all.append(spot_ST)
        self.spot_ST_all = np.array(spot_ST_all)

        # 新
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)

        # print('spot_ST_all',self.spot_ST_all.shape)
        self.spot_ST_all = np.reshape(self.spot_ST_all, (self.spot_ST_all.shape[0]*(all_gene//gene_num),gene_num, self.spot_ST_all.shape[2],self.spot_ST_all.shape[3]))
        # print('spot_ST_all',self.spot_ST_all.shape)
        for ii in range(self.spot_ST_all.shape[0]):
            for jj in range(self.spot_ST_all.shape[1]):
                if np.sum(self.spot_ST_all[ii, jj])!= 0:
                    Max = np.max(self.spot_ST_all[ii, jj])
                    Min = np.min(self.spot_ST_all[ii, jj])
                    self.spot_ST_all[ii, jj] = (self.spot_ST_all[ii, jj] - Min) / (Max - Min)

        ### WSI 5120
        WSI_5120_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'NBME/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_5120 = np.load(data_root + 'NBME/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
                max = np.max(WSI_5120)
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
        self.WSI_5120_all = np.array(WSI_5120_all)
             # 对WSI 5120数据按照分组数量复制到batch维度
        self.WSI_5120_all = []
        for _ in range(all_gene//gene_num):
            self.WSI_5120_all.append(self.WSI_5120_all)
        self.WSI_5120_all = np.concatenate(self.WSI_5120_all, axis=0)

        ### WSI 320
        self.num_320 = []
        WSI_320_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'NBME/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_320 = np.load(data_root + 'NBME/WSI/extract/' + sample_id + '/' + patch_id + '/320_to16.npy')
                WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
                WSI_320_all.append(WSI_320)
        self.WSI_320_all = np.array(WSI_320_all)

        # 对WSI 320数据按照分组数量复制到batch维度
        self.WSI_320_all = []
        for _ in range(all_gene//gene_num):
            self.WSI_320_all.append(self.WSI_320_all)
        self.WSI_320_all = np.concatenate(self.WSI_320_all, axis=0)
        max_320 = np.max(WSI_320)
        a = 1
    def __len__(self):
        # 返回处理后数据的batch维度大小（也就是分组后的数量）
        return self.spot_ST_all.shape[0]

    def __getitem__(self, index):
        '''
            返回对应索引位置处理后的各项数据，包含分组融入batch维度后的ST数据以及对应复制后的WSI数据。
        '''
        return self.spot_ST_all[index], self.WSI_5120_all[index], self.WSI_320_all[index]


if  __name__ == '__main__':
    import cv2
    data_root = '/home/hanyu/MIA/data/'
    dataset_use = 'Xenium'#BreastST,SGE
    status = 'Test'
    SR_times = 10
    gene_num = 10
    all_gene=30
    dataset = load_data(data_root,dataset_use,status,SR_times,gene_num,all_gene)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True
    )
    for idx,(SR_ST_all,LR_ST_all, WSI_5120_all,WSI_320_all, gene_captions, Gene_index_map) in enumerate(dataloader):
        SR_ST_all = SR_ST_all.numpy()
        SR_ST = SR_ST_all[0]
        LR_ST_all = LR_ST_all.numpy()
        LR_ST = LR_ST_all[0]
        WSI_5120_all = WSI_5120_all.numpy()
        WSI = WSI_5120_all[0]
        WSI = np.transpose(WSI, axes=(1, 2, 0))
        # WSI_5120_all = WSI_5120_all.numpy()
        wsi_path = f'temp4/{idx+1}_gene_WSI.png'
        plt.imsave(wsi_path, WSI)
        for k in range(SR_ST.shape[0]):
            gt_path = f'temp4/{idx+1}_gene_{k+1}_GT.png'
            lr_path = f'temp4/{idx+1}_gene_{k+1}_LR.png'
            # pred_path = f'temp/patch_{idx+1}_gene_{k+1}.png'
            plt.imsave(gt_path, SR_ST[k], cmap='viridis')
            # 对LR_ST[k]进行分辨率放大
            LR_ST_resized = cv2.resize(LR_ST[k], (256, 256), interpolation=cv2.INTER_LINEAR)
            plt.imsave(lr_path, LR_ST_resized, cmap='viridis')
            # plt.imsave(pred_path, pred_sample[k], cmap='viridis')
            


