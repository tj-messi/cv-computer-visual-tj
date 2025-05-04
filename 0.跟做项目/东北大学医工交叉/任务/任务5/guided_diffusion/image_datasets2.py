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
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
def load_data(data_root,dataset_use,status,SR_times,gene_num,all_gene,gene_order=None,gene_name_order=None
):

    if dataset_use=='Xenium':
        dataset= Xenium_dataset(data_root,SR_times,status,gene_num,all_gene,gene_order,gene_name_order)
    elif dataset_use=='SGE':
        dataset= SGE_dataset(data_root,gene_num,all_gene)
    elif dataset_use=='BreastST':
        dataset= BreastST_dataset(data_root,gene_num,all_gene)
    return dataset

class Xenium_dataset(Dataset):
    def __init__(self, data_root, SR_times, status, gene_num, all_gene, gene_order, gene_name_order):
        """
        data_root: 数据根目录的路径。
        SR_times: 下采样倍数，影响加载的HR ST数据的分辨率。
        status: 指定数据集的状态，值为 'Train' 或 'Test'。
        gene_num: 每个样本需要处理的基因数量（如20）。
        all_gene: 总基因数量（当传入完整的基因序列时使用）。
        gene_order: 若传入的为每组的基因索引（子集，长度应为gene_num），则直接使用；否则从文件加载完整排序。
        gene_name_order: 基因名称列表或路径
        """
        if status == 'Train':
            sample_name = ['20231115pancreascancer', '20240514pancreascancer']
        elif status == 'Test':
            sample_name = ['20240319pancreascancer']
        
 

        self.gene_order = gene_order 
        self.gene_name_order = gene_name_order
        SR_ST_all = []
        self.gene_scale = []  # 保存每个patch对应的基因索引组
        self.gene_num = gene_num
        
        # 加载HR ST数据
        for sample_id in sample_name:
            sub_patches = os.listdir(os.path.join(data_root, 'HR_ST/extract', sample_id))
            for patch_id in sub_patches:
                if SR_times == 10:
                    SR_ST = sp_sparse.load_npz(os.path.join(data_root, 'HR_ST/extract', sample_id, patch_id, 'HR_ST_256.npz')).toarray().reshape(256, 256, 98)
                elif SR_times == 5:
                    SR_ST = sp_sparse.load_npz(os.path.join(data_root, 'HR_ST/extract', sample_id, patch_id, 'HR_ST_128.npz')).toarray().reshape(128, 128, 98)
                SR_ST = np.transpose(SR_ST, axes=(2, 0, 1))
                SR_ST_all.append(SR_ST)
                # 对于每个patch，使用传入的gene_order作为当前的基因组
                self.gene_scale.append(self.gene_order)
        self.SR_ST_all = np.array(SR_ST_all)
        # 直接将gene_scale转换为数组，不做额外重组
        self.gene_scale_groups = np.array(self.gene_scale)
        # 仅选择传入的基因（子集）对应的通道
        self.SR_ST_all = self.SR_ST_all[:, self.gene_order, ...].astype(np.float64)
        # 如果每个patch仅对应一组基因，则直接赋值
        self.SR_ST_all_groups = self.SR_ST_all

        # 对HR ST数据归一化
        for ii in range(self.SR_ST_all_groups.shape[0]):
            for jj in range(self.SR_ST_all_groups.shape[1]):
                if np.sum(self.SR_ST_all_groups[ii, jj]) != 0:
                    Max = np.max(self.SR_ST_all_groups[ii, jj])
                    Min = np.min(self.SR_ST_all_groups[ii, jj])
                    self.SR_ST_all_groups[ii, jj] = (self.SR_ST_all_groups[ii, jj] - Min) / (Max - Min)
        
        # 加载spot ST数据
        spot_ST_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(os.path.join(data_root, 'spot_ST/extract', sample_id))
            for patch_id in sub_patches:
                spot_ST = sp_sparse.load_npz(os.path.join(data_root, 'spot_ST/extract', sample_id, patch_id, 'spot_ST.npz')).toarray().reshape(26, 26, 98)
                spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
                spot_ST_all.append(spot_ST)
        self.spot_ST_all = np.array(spot_ST_all)
        self.spot_ST_all = self.spot_ST_all[:, self.gene_order, ...].astype(np.float64)
        self.spot_ST_all_groups = self.spot_ST_all

        # 对spot ST数据归一化
        for ii in range(self.spot_ST_all_groups.shape[0]):
            for jj in range(self.spot_ST_all_groups.shape[1]):
                if np.sum(self.spot_ST_all_groups[ii, jj]) != 0:
                    Max = np.max(self.spot_ST_all_groups[ii, jj])
                    Min = np.min(self.spot_ST_all_groups[ii, jj])
                    self.spot_ST_all_groups[ii, jj] = (self.spot_ST_all_groups[ii, jj] - Min) / (Max - Min)
        
        # 加载WSI 5120数据
        WSI_5120_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(os.path.join(data_root, 'WSI/extract', sample_id))
            for patch_id in sub_patches:
                WSI_5120 = np.load(os.path.join(data_root, 'WSI/extract', sample_id, patch_id, '5120_to256.npy'))
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
        self.WSI_5120_all = np.array(WSI_5120_all)
        # 这里不需要复制多份，因为每个patch只对应一组基因
        self.WSI_5120_all_expanded = self.WSI_5120_all
        
        # 加载WSI 320数据
        WSI_320_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(os.path.join(data_root, 'WSI/extract', sample_id))
            for patch_id in sub_patches:
                WSI_320 = np.load(os.path.join(data_root, 'WSI/extract', sample_id, patch_id, '320_to16.npy'))
                WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
                WSI_320_all.append(WSI_320)
        self.WSI_320_all = np.array(WSI_320_all)
        self.WSI_320_all_expanded = self.WSI_320_all
        
        # 加载BERT模型，提取基因名称特征
        self.tokenizer = AutoTokenizer.from_pretrained("/home/hanyu/hanyu_code/NC-code3.13/bert", local_files_only=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained("/home/hanyu/hanyu_code/NC-code3.13/bert", local_files_only=True, trust_remote_code=True)
        self.model.eval()
        
        if isinstance(gene_name_order, str) and os.path.exists(gene_name_order):
            with open(gene_name_order, "r") as f:
                gene_names = [f"This gene’s name is called {line.strip()}." for line in f if line.strip()]
        else:
            gene_names = [f"This gene’s name is called {gene}." for gene in gene_name_order]
        print(gene_names[:5])
        
        gene_feats = []
        with torch.no_grad():
            for gene in gene_names:
                inputs = self.tokenizer(gene, return_tensors="pt")
                outputs = self.model(**inputs)
                gene_embedding = outputs.pooler_output.squeeze(0)
                gene_feats.append(gene_embedding)
        self.gene_name_features = torch.stack(gene_feats, dim=0)
        print('gene_name_features shape:', self.gene_name_features.shape)
        
        metadata_prompt = "Provide spatial transcriptomics data from the Xenium platform for human species, with a cancer condition, and breast tissue type."
        with torch.no_grad():
            inputs = self.tokenizer(metadata_prompt, return_tensors="pt")
            outputs = self.model(**inputs)
            self.metadata_feature = outputs.pooler_output.squeeze(0)
        print('metadata_feature shape:', self.metadata_feature.shape)
    
    def __len__(self):
        return self.SR_ST_all_groups.shape[0]
    
    def __getitem__(self, index):
        # 获取对应patch的基因索引组
        gene_class = self.gene_scale_groups[index]
        Gene_index_maps = []
        for gene_code in gene_class:
            Gene_codes = gray_value_of_gene(gene_code, self.gene_order)
            Gene_index_map = np.ones((256, 256, 1)) * Gene_codes / 255.0
            Gene_index_maps.append(Gene_index_map)
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2)
        final_Gene_index_map = np.moveaxis(final_Gene_index_map, 2, 0)
        return (self.SR_ST_all_groups[index],
                self.spot_ST_all_groups[index],
                self.WSI_5120_all_expanded[index],
                self.WSI_320_all_expanded[index],
                gene_class,
                final_Gene_index_map,
                self.gene_name_features,
                self.metadata_feature)




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

        print('spot_ST_all',self.spot_ST_all.shape)
        self.spot_ST_all_groups = np.reshape(self.spot_ST_all, (self.spot_ST_all.shape[0]*(all_gene//gene_num),gene_num, self.spot_ST_all.shape[2],self.spot_ST_all.shape[3]))
        print('spot_ST_all_groups',self.spot_ST_all_groups.shape)
        for ii in range(self.spot_ST_all_groups.shape[0]):
            for jj in range(self.spot_ST_all_groups.shape[1]):
                if np.sum(self.spot_ST_all_groups[ii, jj])!= 0:
                    Max = np.max(self.spot_ST_all_groups[ii, jj])
                    Min = np.min(self.spot_ST_all_groups[ii, jj])
                    self.spot_ST_all_groups[ii, jj] = (self.spot_ST_all_groups[ii, jj] - Min) / (Max - Min)

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
        self.WSI_5120_all_expanded = []
        for _ in range(all_gene//gene_num):
            self.WSI_5120_all_expanded.append(self.WSI_5120_all)
        self.WSI_5120_all_expanded = np.concatenate(self.WSI_5120_all_expanded, axis=0)

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
        self.WSI_320_all_expanded = []
        for _ in range(all_gene//gene_num):
            self.WSI_320_all_expanded.append(self.WSI_320_all)
        self.WSI_320_all_expanded = np.concatenate(self.WSI_320_all_expanded, axis=0)
        max_320 = np.max(WSI_320)
        a = 1
    def __len__(self):
        # 返回处理后数据的batch维度大小（也就是分组后的数量）
        return self.spot_ST_all_groups.shape[0]

    def __getitem__(self, index):
        '''
            返回对应索引位置处理后的各项数据，包含分组融入batch维度后的ST数据以及对应复制后的WSI数据。
        '''
        return self.spot_ST_all_groups[index], self.WSI_5120_all_expanded[index], self.WSI_320_all_expanded[index]


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

        print('spot_ST_all',self.spot_ST_all.shape)
        self.spot_ST_all_groups = np.reshape(self.spot_ST_all, (self.spot_ST_all.shape[0]*(all_gene//gene_num),gene_num, self.spot_ST_all.shape[2],self.spot_ST_all.shape[3]))
        print('spot_ST_all_groups',self.spot_ST_all_groups.shape)
        for ii in range(self.spot_ST_all_groups.shape[0]):
            for jj in range(self.spot_ST_all_groups.shape[1]):
                if np.sum(self.spot_ST_all_groups[ii, jj])!= 0:
                    Max = np.max(self.spot_ST_all_groups[ii, jj])
                    Min = np.min(self.spot_ST_all_groups[ii, jj])
                    self.spot_ST_all_groups[ii, jj] = (self.spot_ST_all_groups[ii, jj] - Min) / (Max - Min)

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
        self.WSI_5120_all_expanded = []
        for _ in range(all_gene//gene_num):
            self.WSI_5120_all_expanded.append(self.WSI_5120_all)
        self.WSI_5120_all_expanded = np.concatenate(self.WSI_5120_all_expanded, axis=0)

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
        self.WSI_320_all_expanded = []
        for _ in range(all_gene//gene_num):
            self.WSI_320_all_expanded.append(self.WSI_320_all)
        self.WSI_320_all_expanded = np.concatenate(self.WSI_320_all_expanded, axis=0)
        max_320 = np.max(WSI_320)
        a = 1
    def __len__(self):
        # 返回处理后数据的batch维度大小（也就是分组后的数量）
        return self.spot_ST_all_groups.shape[0]

    def __getitem__(self, index):
        '''
            返回对应索引位置处理后的各项数据，包含分组融入batch维度后的ST数据以及对应复制后的WSI数据。
        '''
        return self.spot_ST_all_groups[index], self.WSI_5120_all_expanded[index], self.WSI_320_all_expanded[index]


if  __name__ == '__main__':
    import cv2
    data_root = '/home/zeiler/ST_proj/data/Breast_cancer/'
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
            


