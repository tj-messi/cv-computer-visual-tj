import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
from skimage.measure import shannon_entropy
from PIL import Image,ImageOps
import scipy.sparse as sp_sparse
from transformers import AutoTokenizer, AutoModel
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
    elif 'Xenium5k_humanbreast' in dataset_use:
        dataset= Xenium5khumanbreast(data_root,dataset_use,SR_times,status,gene_num,all_gene,gene_order,gene_name_order)
    elif 'Xenium5k_cervixcervicalcancer' in dataset_use:
        dataset= Xenium5kcervixcancer(data_root,dataset_use,SR_times,status,gene_num,all_gene,gene_order,gene_name_order)  
    elif 'Visiumhdmousebrain4_8' in dataset_use:
       dataset= Xenium5k2_brain(data_root,dataset_use,SR_times,status,gene_num,all_gene,gene_order,gene_name_order)
    elif 'Xenium5k_11111' in dataset_use:
       dataset= Xenium5k(data_root,dataset_use,SR_times,status,gene_num,all_gene,gene_order)
    elif 'VisiumHD_mouseembryo_sorted_data1' in dataset_use:
       dataset= Xenium5k2(data_root,dataset_use,SR_times,status,gene_num,all_gene,gene_order,gene_name_order)
    elif 'Visiumhd_mouse_kidney' in dataset_use:
       dataset= Visiumhdmousekidney(data_root,dataset_use,SR_times,status,gene_num,all_gene,gene_order,gene_name_order)   
    elif dataset_use=='Human-brain-glioblastomamultiforme':
        dataset=Xenium_humanbrain(data_root,dataset_use,SR_times,status,gene_num,all_gene,gene_order)
    elif 'Visiumhd_human_Tonsil' in dataset_use:
       dataset= VisiumhdhumanTonsil(data_root,dataset_use,SR_times,status,gene_num,all_gene,gene_order,gene_name_order)
    elif dataset_use=='visium_mouse_kidney':
        dataset=Xenium5k7(data_root,dataset_use,SR_times,status,gene_num,all_gene,gene_order,gene_name_order)
    elif 'Xenium5k_frozenmousebrain' in dataset_use:
        dataset= Xenium5kfrozenmousebrain(data_root,dataset_use,SR_times,status,gene_num,all_gene,gene_order,gene_name_order)
    return dataset
class Xenium5kfrozenmousebrain(Dataset):
    def __init__(self, data_root, dataset_use, SR_times, status, gene_num, all_gene, gene_order, gene_name_order):
        from transformers import AutoTokenizer, AutoModel
        import os
        import numpy as np
        import scipy.sparse as sp_sparse
        import torch

        if dataset_use == 'Xenium5k_frozenmousebrain':
            sample_name = ['20240801mousebrain']
        else:
            raise ValueError('Invalid dataset_use')

        self.selected_patches = []
        self.gene_order = gene_order
        self.gene_name_order = gene_name_order
        self.gene_num = gene_num

        for sample_id in sample_name:
            # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
                base_path = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/"
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
                        if status == 'Train' and (train_max_row == -1 or b <= train_max_row):
                            # print('patch_id',patch_id)
                            self.selected_patches.append((sample_id, patch_id))
                        # elif status == 'Test' and (train_max_row == -1 or b > train_max_row):
                        #     self.selected_patches.append((sample_id, patch_id))
                        if status == 'Test' :
                            self.selected_patches.append((sample_id, patch_id))
                    except ValueError:
                        continue  # 跳过无法转换为整数的情况

        SR_ST_all, spot_ST_all, WSI_5120_all, WSI_320_all = [], [], [], []
        self.gene_scale = []

        if status == 'Test':
            self.selected_patches.sort(key=lambda x: (int(x[1].split('_')[0]), int(x[1].split('_')[1])))

        for sample_id, patch_id in self.selected_patches:
            hr_path = os.path.join(data_root, dataset_use, 'HR_ST', 'extract', sample_id, patch_id, 'HR_ST_256.npz')
            spot_path = os.path.join(data_root, dataset_use, 'spot_ST', 'extract', sample_id, patch_id, 'spot_ST.npz')

            sr_sparse = sp_sparse.load_npz(hr_path)  # shape: (256*256, all_gene)
            sr_sparse = sr_sparse[:, gene_order]     # shape: (256*256, gene_num)
            SR_ST = sr_sparse.toarray().reshape(256, 256, -1).transpose(2, 0, 1)
            SR_ST_all.append(SR_ST)
            self.gene_scale.append(gene_order)

            spot_sparse = sp_sparse.load_npz(spot_path)  # shape: (26*26, all_gene)
            spot_sparse = spot_sparse[:, gene_order]
            spot_ST = spot_sparse.toarray().reshape(26, 26, -1).transpose(2, 0, 1)
            spot_ST_all.append(spot_ST)

            wsi_path = os.path.join(data_root, dataset_use, 'WSI', 'extract', sample_id, patch_id, '5120_to256.npy')
            WSI_5120 = np.load(wsi_path).transpose(2, 0, 1)
            WSI_5120_all.append(WSI_5120)

            wsi_320_path = os.path.join(data_root, dataset_use, 'WSI', 'extract', sample_id, patch_id, '320_to16.npy')
            WSI_320 = np.load(wsi_320_path).transpose(0, 3, 1, 2)
            WSI_320_all.append(WSI_320)

        self.SR_ST_all = np.array(SR_ST_all, dtype=np.float32)
        self.spot_ST_all = np.array(spot_ST_all, dtype=np.float32)
        self.WSI_5120_all = np.array(WSI_5120_all, dtype=np.float32)
        self.WSI_320_all = np.array(WSI_320_all, dtype=np.float32)
        self.gene_scale = np.array(self.gene_scale)

        for data_arr in [self.SR_ST_all, self.spot_ST_all]:
            for ii in range(data_arr.shape[0]):
                for jj in range(data_arr.shape[1]):
                    if np.sum(data_arr[ii, jj]) != 0:
                        Min, Max = data_arr[ii, jj].min(), data_arr[ii, jj].max()
                        data_arr[ii, jj] = (data_arr[ii, jj] - Min) / (Max - Min + 1e-8)

        self.tokenizer = AutoTokenizer.from_pretrained("/media/cbtil/T7 Shield/NMI/bert", local_files_only=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained("/media/cbtil/T7 Shield/NMI/bert", local_files_only=True, trust_remote_code=True)
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

        metadata_prompt = ("Provide spatial transcriptomics data from the Xenium5k platform for mouse species, with a cancer condition, and brain tissue type.")
        with torch.no_grad():
            inputs = self.tokenizer(metadata_prompt, return_tensors="pt")
            outputs = self.model(**inputs)
            self.metadata_feature = outputs.pooler_output.squeeze(0)
        print('metadata_feature shape:', self.metadata_feature.shape)
        print(f"Test模式下，选中的patch数量：{len(self.selected_patches)}")

    def __len__(self):
        return len(self.selected_patches)

    def __getitem__(self, index):
        gene_class = self.gene_scale[index]
        Gene_index_maps = [
            np.ones((256, 256, 1)) * gray_value_of_gene(code, self.gene_order) / 255.0
            for code in gene_class
        ]
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2).transpose(2, 0, 1)

        return (
            self.SR_ST_all[index],
            self.spot_ST_all[index],
            self.WSI_5120_all[index],
            self.WSI_320_all[index],
            gene_class,
            final_Gene_index_map,
            self.gene_name_features,
            self.metadata_feature
        )
class Xenium5kcervixcancer(Dataset):
    def __init__(self, data_root, dataset_use, SR_times, status, gene_num, all_gene, gene_order, gene_name_order):
        from transformers import AutoTokenizer, AutoModel
        import os
        import numpy as np
        import scipy.sparse as sp_sparse
        import torch

        if dataset_use == 'Xenium5k_cervixcervicalcancer':
            sample_name = ['20240905cervix']
        else:
            raise ValueError('Invalid dataset_use')

        self.selected_patches = []
        self.gene_order = gene_order
        self.gene_name_order = gene_name_order
        self.gene_num = gene_num

        for sample_id in sample_name:
                # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
                base_path = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/"
                print('base_path', base_path)
                sub_patches = os.listdir(base_path)
                
                for patch_id in sub_patches:
                    parts = patch_id.split('_')
                    if len(parts) < 2:
                        continue  # 跳过非法格式
                    
                    try:
                        a = int(parts[0])  # 提取行号
                        b = int(parts[1])  # 提取列号
                        
                        # 根据行号划分训练集和测试集
                        # 行号0-3作为训练集，行号4+作为测试集
                        if status == 'Train' and 0 <= a <= 4:
                            self.selected_patches.append((sample_id, patch_id))
                        #elif status == 'Test' and a >= 5:
                        elif status == 'Test' :
                            self.selected_patches.append((sample_id, patch_id))
                            
                    except ValueError:
                        continue  # 跳过无法转换为整数的情况

        SR_ST_all, spot_ST_all, WSI_5120_all, WSI_320_all = [], [], [], []
        self.gene_scale = []

        if status == 'Test':
            self.selected_patches.sort(key=lambda x: (int(x[1].split('_')[0]), int(x[1].split('_')[1])))

        for sample_id, patch_id in self.selected_patches:
            hr_path = os.path.join(data_root, dataset_use, 'HR_ST', 'extract', sample_id, patch_id, 'HR_ST_256.npz')
            spot_path = os.path.join(data_root, dataset_use, 'spot_ST', 'extract', sample_id, patch_id, 'spot_ST.npz')

            sr_sparse = sp_sparse.load_npz(hr_path)  # shape: (256*256, all_gene)
            sr_sparse = sr_sparse[:, gene_order]     # shape: (256*256, gene_num)
            SR_ST = sr_sparse.toarray().reshape(256, 256, -1).transpose(2, 0, 1)
            SR_ST_all.append(SR_ST)
            self.gene_scale.append(gene_order)

            spot_sparse = sp_sparse.load_npz(spot_path)  # shape: (26*26, all_gene)
            spot_sparse = spot_sparse[:, gene_order]
            spot_ST = spot_sparse.toarray().reshape(26, 26, -1).transpose(2, 0, 1)
            spot_ST_all.append(spot_ST)

            wsi_path = os.path.join(data_root, dataset_use, 'WSI', 'extract', sample_id, patch_id, '5120_to256.npy')
            WSI_5120 = np.load(wsi_path).transpose(2, 0, 1)
            WSI_5120_all.append(WSI_5120)

            wsi_320_path = os.path.join(data_root, dataset_use, 'WSI', 'extract', sample_id, patch_id, '320_to16.npy')
            WSI_320 = np.load(wsi_320_path).transpose(0, 3, 1, 2)
            WSI_320_all.append(WSI_320)

        self.SR_ST_all = np.array(SR_ST_all, dtype=np.float32)
        self.spot_ST_all = np.array(spot_ST_all, dtype=np.float32)
        self.WSI_5120_all = np.array(WSI_5120_all, dtype=np.float32)
        self.WSI_320_all = np.array(WSI_320_all, dtype=np.float32)
        self.gene_scale = np.array(self.gene_scale)

        for data_arr in [self.SR_ST_all, self.spot_ST_all]:
            for ii in range(data_arr.shape[0]):
                for jj in range(data_arr.shape[1]):
                    if np.sum(data_arr[ii, jj]) != 0:
                        Min, Max = data_arr[ii, jj].min(), data_arr[ii, jj].max()
                        data_arr[ii, jj] = (data_arr[ii, jj] - Min) / (Max - Min + 1e-8)

        self.tokenizer = AutoTokenizer.from_pretrained("/media/cbtil/T7 Shield/NMI/bert", local_files_only=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained("/media/cbtil/T7 Shield/NMI/bert", local_files_only=True, trust_remote_code=True)
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

        metadata_prompt = ("Provide spatial transcriptomics data from the Xenium5k platform for human species, with a cancer condition, and cervix tissue type.")
        with torch.no_grad():
            inputs = self.tokenizer(metadata_prompt, return_tensors="pt")
            outputs = self.model(**inputs)
            self.metadata_feature = outputs.pooler_output.squeeze(0)
        print('metadata_feature shape:', self.metadata_feature.shape)
        print(f"Test模式下，选中的patch数量：{len(self.selected_patches)}")

    def __len__(self):
        return len(self.selected_patches)

    def __getitem__(self, index):
        gene_class = self.gene_scale[index]
        Gene_index_maps = [
            np.ones((256, 256, 1)) * gray_value_of_gene(code, self.gene_order) / 255.0
            for code in gene_class
        ]
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2).transpose(2, 0, 1)

        return (
            self.SR_ST_all[index],
            self.spot_ST_all[index],
            self.WSI_5120_all[index],
            self.WSI_320_all[index],
            gene_class,
            final_Gene_index_map,
            self.gene_name_features,
            self.metadata_feature
        )
def normalize_patches(tensor: torch.Tensor, l1_scale: float = 1e4):
    """
    对输入的 4D 张量做规范化：
      1. L1 归一化 + log1p 缩放
      2. 按通道最大值做 0-1 归一化，并替换 NaN/inf
    Args:
      tensor: torch.Tensor, shape=(N, C, H, W)
      l1_scale: float, 用于 log1p 前的缩放因子
    Returns:
      normalized: torch.Tensor, shape=(N, C, H, W)，0-1 归一化结果
      coefs: torch.Tensor, shape=(N,)，每个样本的恢复系数
    """
    N, C, H, W = tensor.shape

    # 1. L1 归一化 + log1p
    flat = tensor.view(N, -1)
    l1_norm = flat.abs().sum(dim=1, keepdim=True)
    l1_norm = torch.where(l1_norm == 0, torch.ones_like(l1_norm), l1_norm)
    flat_norm = flat / l1_norm
    flat_norm = torch.log1p(flat_norm * l1_scale)
    patches = flat_norm.view(N, C, H, W)

    # 2. 基于通道最大值 0-1 归一化
    channels_max = patches.max(dim=1, keepdim=True)[0]  # (N,1,H,W)
    normed = patches / (channels_max + 1e-12)
    normed = torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. 计算恢复系数 coefs
    max_vals = channels_max.view(N, -1).max(dim=1)[0]
    global_max = max_vals.max()
    coefs = max_vals / (global_max + 1e-12)

    return normed, coefs
class Xenium5khumanbreast(Dataset):
    def __init__(self, data_root, dataset_use, SR_times, status, gene_num, all_gene, gene_order, gene_name_order):
        from transformers import AutoTokenizer, AutoModel
        import os
        import numpy as np
        import scipy.sparse as sp_sparse
        import torch

        if dataset_use == 'Xenium5k_humanbreast':
            sample_name = ['20241024Breast']
        else:
            raise ValueError('Invalid dataset_use')

        self.selected_patches = []
        self.gene_order = gene_order
        self.gene_name_order = gene_name_order
        self.gene_num = gene_num

        for sample_id in sample_name:
            base_path = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/"
            print('base_path', base_path)
            sub_patches = os.listdir(base_path)
            for patch_id in sub_patches:
                parts = patch_id.split('_')
                if len(parts) < 2:
                    continue
                try:
                    a = int(parts[0])
                    b = int(parts[1])
                    if status == 'Train' and 0 <= a <= 4:
                        self.selected_patches.append((sample_id, patch_id))
                    elif status == 'Test':
                        self.selected_patches.append((sample_id, patch_id))
                except ValueError:
                    continue

        SR_ST_all, spot_ST_all, WSI_5120_all, WSI_320_all = [], [], [], []
        self.gene_scale = []

        if status == 'Test':
            self.selected_patches.sort(key=lambda x: (int(x[1].split('_')[0]), int(x[1].split('_')[1])))

        for sample_id, patch_id in self.selected_patches:
            hr_path = os.path.join(data_root, dataset_use, 'HR_ST', 'extract', sample_id, patch_id, 'HR_ST_256.npz')
            spot_path = os.path.join(data_root, dataset_use, 'spot_ST', 'extract', sample_id, patch_id, 'spot_ST.npz')

            sr_sparse = sp_sparse.load_npz(hr_path)  # shape: (256*256, all_gene)
            sr_sparse = sr_sparse[:, gene_order]     # shape: (256*256, gene_num)
            SR_ST = sr_sparse.toarray().reshape(256, 256, -1).transpose(2, 0, 1)
            SR_ST_all.append(SR_ST)
            self.gene_scale.append(gene_order)

            spot_sparse = sp_sparse.load_npz(spot_path)  # shape: (26*26, all_gene)
            spot_sparse = spot_sparse[:, gene_order]
            spot_ST = spot_sparse.toarray().reshape(26, 26, -1).transpose(2, 0, 1)
            spot_ST_all.append(spot_ST)

            wsi_path = os.path.join(data_root, dataset_use, 'WSI', 'extract', sample_id, patch_id, '5120_to256.npy')
            WSI_5120 = np.load(wsi_path).transpose(2, 0, 1)
            WSI_5120_all.append(WSI_5120)

            wsi_320_path = os.path.join(data_root, dataset_use, 'WSI', 'extract', sample_id, patch_id, '320_to16.npy')
            WSI_320 = np.load(wsi_320_path).transpose(0, 3, 1, 2)
            WSI_320_all.append(WSI_320)

        self.SR_ST_all = np.array(SR_ST_all, dtype=np.float32)
        self.spot_ST_all = np.array(spot_ST_all, dtype=np.float32)
        self.WSI_5120_all = np.array(WSI_5120_all, dtype=np.float32)
        self.WSI_320_all = np.array(WSI_320_all, dtype=np.float32)
        self.gene_scale = np.array(self.gene_scale)

        # 归一化 SR_ST: L1+log + channel max-min to 0-1
        sr_tensor = torch.from_numpy(self.SR_ST_all.astype(np.float32))
        normed, coefs = normalize_patches(sr_tensor)
        self.SR_ST_all_groups = normed.numpy()
        self.SR_ST_coefs = coefs.numpy()

        for data_arr in [ self.spot_ST_all]:
            for ii in range(data_arr.shape[0]):
                for jj in range(data_arr.shape[1]):
                    if np.sum(data_arr[ii, jj]) != 0:
                        Min, Max = data_arr[ii, jj].min(), data_arr[ii, jj].max()
                        data_arr[ii, jj] = (data_arr[ii, jj] - Min) / (Max - Min + 1e-8)

        self.tokenizer = AutoTokenizer.from_pretrained("/media/cbtil/T7 Shield/NMI/bert", local_files_only=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained("/media/cbtil/T7 Shield/NMI/bert", local_files_only=True, trust_remote_code=True)
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

        metadata_prompt = ("Provide spatial transcriptomics data from the Xenium5k platform for human species, with a cancer condition, and breast tissue type.")
        with torch.no_grad():
            inputs = self.tokenizer(metadata_prompt, return_tensors="pt")
            outputs = self.model(**inputs)
            self.metadata_feature = outputs.pooler_output.squeeze(0)
        print('metadata_feature shape:', self.metadata_feature.shape)
        print(f"Test模式下，选中的patch数量：{len(self.selected_patches)}")

    def __len__(self):
        return len(self.selected_patches)

    def __getitem__(self, index):
        gene_class = self.gene_scale[index]
        Gene_index_maps = [ 
            np.ones((256, 256, 1)) * gray_value_of_gene(code, self.gene_order) / 255.0
            for code in gene_class
        ]
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2).transpose(2, 0, 1)

        return (
            self.SR_ST_all[index],
            self.spot_ST_all[index],
            self.WSI_5120_all[index],
            self.WSI_320_all[index],
            gene_class,
            final_Gene_index_map,
            self.gene_name_features,
            self.metadata_feature
        )


class Xenium5k7(Dataset):
    def __init__(self, data_root, dataset_use,SR_times, status, gene_num, all_gene, gene_order,gene_name_order):
        '''
            data_root: 数据根目录路径
            SR_times: 下采样倍数
            status: 数据集状态 'Train' 或 'Test'
            gene_num: 基因数量
            all_gene: 总基因数（用于reshape）
            gene_order: 基因顺序索引
        '''

        if dataset_use == 'visium_mouse_kidney':
            sample_name = ['20210816']
        else:
            raise ValueError('Invalid dataset_use')

        self.selected_patches = []  # 存储符合条件的样本路径
        self.gene_order = gene_order
        self.gene_name_order = gene_name_order
        self.gene_num = gene_num
        # 根据status筛选符合条件的patch

        if dataset_use == 'visium_mouse_kidney':
            for sample_id in sample_name:
                base_path = f"{data_root}{dataset_use}/spot_ST/extract/{sample_id}"
                print('base_path', base_path)
                sub_patches = os.listdir(base_path)
                
                if status == 'Test':
                    for patch_id in sub_patches:
                        self.selected_patches.append((sample_id, patch_id))

            # 初始化数据容器
            spot_ST_all, WSI_5120_all,WSI_320_all = [], [],[]
            self.gene_scale = []

            # 有序输出
            if status == 'Test':
                self.selected_patches.sort(key=lambda x: (int(x[1].split('_')[0]), int(x[1].split('_')[1])))
        #print('self.selected_patches',self.selected_patches)
        # print(self.selected_patches)
        # 统一加载逻辑（确保三个数据部分顺序一致）
        for sample_id, patch_id in self.selected_patches:
            spot_ST = sp_sparse.load_npz(os.path.join(data_root,dataset_use, 'spot_ST/extract/',sample_id, patch_id, 'spot_ST.npz')).toarray().reshape(26, 26, 19465)
            spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
            spot_ST_all.append(spot_ST)
            wsi_path = f"{data_root}{dataset_use}/WSI/extract/20210816/{patch_id}/5120_to256.npy"
            WSI_5120 = np.load(wsi_path)
            WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
            WSI_5120_all.append(WSI_5120)
            wsi_320_path =  f"{data_root}{dataset_use}/WSI/extract/20210816/{patch_id}/320_to16.npy"
            #print(wsi_320_path)
            WSI_320 = np.load(wsi_320_path)
            WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
            WSI_320_all.append(WSI_320)
            self.gene_scale.append(gene_order)  # 记录基因顺序
       
        # 转换为numpy数组
        
        #print(gene_order)
        #print(self.SR_ST_all.shape)
        self.spot_ST_all = np.array(spot_ST_all).astype(np.float64)
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)
        self.WSI_5120_all = np.array(WSI_5120_all)
        self.WSI_320_all = np.array(WSI_320_all)

        self.gene_scale = np.array(self.gene_scale)

        # 数据标准化（保持原有逻辑）
        for ii in range(self.spot_ST_all.shape[0]):
            for jj in range(self.spot_ST_all.shape[1]):
                if np.sum(self.spot_ST_all[ii, jj]) != 0:
                    Min, Max = self.spot_ST_all[ii, jj].min(), self.spot_ST_all[ii, jj].max()
                    self.spot_ST_all[ii, jj] = (self.spot_ST_all[ii, jj] - Min) / (Max - Min + 1e-8)
     
        #print('self.SR_ST_all',self.SR_ST_all.shape)
        # print('self.spot_ST_all',self.spot_ST_all.shape)
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
        
        metadata_prompt = ("Provide spatial transcriptomics data from the Visium platform for mouse species, with a healthy condition, and kidney tissue type.")
        with torch.no_grad():
            inputs = self.tokenizer(metadata_prompt, return_tensors="pt")
            outputs = self.model(**inputs)
            self.metadata_feature = outputs.pooler_output.squeeze(0)
        print('metadata_feature shape:', self.metadata_feature.shape)
        print(f"Test模式下，选中的patch数量：{len(self.selected_patches)}")
    def __len__(self):
        return len(self.selected_patches)  # 直接以筛选后的样本数为准

    def __getitem__(self, index):
        if index >= len(self.gene_scale):
            raise IndexError(f"Index {index} is out of bounds for self.gene_scale with size {len(self.gene_scale)}")
        # 基因位置编码（保持原有逻辑）
        gene_class = self.gene_scale[index]
        Gene_index_maps = [
            np.ones((256, 256, 1)) * gray_value_of_gene(code, self.gene_order) / 255.0
            for code in gene_class
        ]
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2).transpose(2, 0, 1)
        
        
        return (
            self.spot_ST_all[index],
            self.WSI_5120_all[index],
            self.WSI_320_all[index],
            gene_class,
            final_Gene_index_map,
            self.gene_name_features,   # shape: [gene_num, hidden_size]
            self.metadata_feature 
        )
        
        
class Xenium5k2_brain(Dataset):
    def __init__(self, data_root, dataset_use,SR_times, status, gene_num, all_gene, gene_order,gene_name_order):
        '''
            data_root: 数据根目录路径
            SR_times: 下采样倍数
            status: 数据集状态 'Train' 或 'Test'
            gene_num: 基因数量
            all_gene: 总基因数（用于reshape）
            gene_order: 基因顺序索引
        '''

        if dataset_use == 'Visiumhdmousebrain4_8':
            sample_name = ['20240917mousebrain']
        else:
            raise ValueError('Invalid dataset_use')

        self.selected_patches = []  # 存储符合条件的样本路径
        self.gene_order = gene_order
        self.gene_name_order = gene_name_order
        self.gene_num = gene_num
        # 根据status筛选符合条件的patch

        if dataset_use == 'Visiumhdmousebrain4_8':
                # 根据 patch 名称中下划线分割后的第二部分判断 patch 编号
                
            for sample_id in sample_name:
                # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
                base_path = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}"
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
            # 初始化数据容器
            SR_ST_all, spot_ST_all, WSI_5120_all,WSI_320_all = [], [], [],[]
            self.gene_scale = []

            # 有序输出
            if status == 'Test':
                self.selected_patches.sort(key=lambda x: (int(x[1].split('_')[0]), int(x[1].split('_')[1])))
        #print('self.selected_patches',self.selected_patches)
        # print(self.selected_patches)
        # 统一加载逻辑（确保三个数据部分顺序一致）
        for sample_id, patch_id in self.selected_patches:
            # ------------------- 加载HR_ST -------------------
            hr_st_dir = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/{patch_id}/"
            if SR_times == 10:
                #print(os.path.join(data_root,dataset_use,'HR_ST/extract/', patch_id, 'HR_ST_256.npz'))
                SR_ST = sp_sparse.load_npz(os.path.join(data_root,dataset_use,'HR_ST/extract/',sample_id, patch_id, 'HR_ST_256.npz')).toarray().reshape(256, 256, 8049)
            elif SR_times == 5:
                SR_ST = sp_sparse.load_npz(os.path.join(data_root,dataset_use,'HR_ST/extract/',sample_id, patch_id, 'HR_ST_128.npz')).toarray().reshape(128, 128, 8049)
            SR_ST = np.transpose(SR_ST, axes=(2, 0, 1))
            SR_ST_all.append(SR_ST)
            self.gene_scale.append(gene_order)  # 记录基因顺序

            # ------------------- 加载spot_ST -------------------
            spot_st_path = f"{data_root}{dataset_use}/spot_ST/{patch_id}/spot_ST_1000.npz"
            spot_ST = sp_sparse.load_npz(os.path.join(data_root,dataset_use, 'spot_ST/extract/',sample_id, patch_id, 'spot_ST.npz')).toarray().reshape(26, 26, 8049)
            spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
            spot_ST_all.append(spot_ST)

            # ------------------- 加载WSI -------------------
            wsi_path = f"{data_root}{dataset_use}/WSI/extract/20240917mousebrain/{patch_id}/5120_to256.npy"
            WSI_5120 = np.load(wsi_path)
            WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
            WSI_5120_all.append(WSI_5120)
            wsi_320_path = os.path.join(data_root,dataset_use, "WSI", "extract", sample_id, patch_id, "320_to16.npy")
            #print(wsi_320_path)
            WSI_320 = np.load(wsi_320_path)
            WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
            WSI_320_all.append(WSI_320)
       
        # 转换为numpy数组
        self.SR_ST_all = np.array(SR_ST_all).astype(np.float64)
        #print(gene_order)
        #print(self.SR_ST_all.shape)
        self.SR_ST_all = self.SR_ST_all[:, gene_order, ...].astype(np.float64)
        self.spot_ST_all = np.array(spot_ST_all).astype(np.float64)
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)
        self.WSI_5120_all = np.array(WSI_5120_all)
        self.WSI_320_all = np.array(WSI_320_all)

        self.gene_scale = np.array(self.gene_scale)

        # 数据标准化（保持原有逻辑）
        for data_arr in [self.SR_ST_all, self.spot_ST_all]:
            for ii in range(data_arr.shape[0]):
                for jj in range(data_arr.shape[1]):
                    if np.sum(data_arr[ii, jj]) != 0:
                        Min, Max = data_arr[ii, jj].min(), data_arr[ii, jj].max()
                        data_arr[ii, jj] = (data_arr[ii, jj] - Min) / (Max - Min + 1e-8)
        #print('self.SR_ST_all',self.SR_ST_all.shape)
        # print('self.spot_ST_all',self.spot_ST_all.shape)
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
        
        metadata_prompt = ("Provide spatial transcriptomics data from the VisiumHD platform for mouse species, with a healthy condition, and brain tissue type.")
        with torch.no_grad():
            inputs = self.tokenizer(metadata_prompt, return_tensors="pt")
            outputs = self.model(**inputs)
            self.metadata_feature = outputs.pooler_output.squeeze(0)
        print('metadata_feature shape:', self.metadata_feature.shape)
    def __len__(self):
        return len(self.selected_patches)  # 直接以筛选后的样本数为准

    def __getitem__(self, index):
        # 基因位置编码（保持原有逻辑）
        gene_class = self.gene_scale[index]
        Gene_index_maps = [
            np.ones((256, 256, 1)) * gray_value_of_gene(code, self.gene_order) / 255.0
            for code in gene_class
        ]
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2).transpose(2, 0, 1)
        
        return (
            self.SR_ST_all[index],
            self.spot_ST_all[index],
            self.WSI_5120_all[index],
            self.WSI_320_all[index],
            gene_class,
            final_Gene_index_map,
            self.gene_name_features,   # shape: [gene_num, hidden_size]
            self.metadata_feature 
        )        
        
class Xenium5k(Dataset):
    def __init__(self, data_root, dataset_use,SR_times, status, gene_num, all_gene, gene_order):
        '''
            data_root: 数据根目录路径
            SR_times: 下采样倍数
            status: 数据集状态 'Train' 或 'Test'
            gene_num: 基因数量
            all_gene: 总基因数（用于reshape）
            gene_order: 基因顺序索引
        '''
        if dataset_use == 'Xenium5k_frozenmousebrain':
            sample_name = ['20240801mousebrain']

        elif dataset_use == 'Xenium5k_ovaryovariancancer':
            sample_name = ['20241218ovary']

        elif dataset_use == 'Xenium5k_cervixcervicalcancer':
            sample_name = ['20240905cervix']
            
        elif dataset_use == 'Xenium5k_human':
            sample_name = ['20241024Breast']
        elif dataset_use == 'Xenium5k_brain-glioblastomamultiforme':
            sample_name = ['20240416Brain']
        elif dataset_use == 'VisiumHD_moueembryo_sorted_data1':
            sample_name = ['20240611mouseembryo']
        else:
            raise ValueError('Invalid dataset_use')

        self.selected_patches = []  # 存储符合条件的样本路径
        self.gene_order = gene_order
        # 根据status筛选符合条件的patch
        if dataset_use == 'Xenium5k_ovaryovariancancer':
            max_col = 3  # 列范围 0-3
            
            for sample_id in sample_name:
                base_path = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/"
                print('base_path', base_path)
                sub_patches = os.listdir(base_path)
                
                for patch_id in sub_patches:
                    parts = patch_id.split('_')
                    if len(parts) < 2:
                        continue  # 跳过非法格式
                        
                    try:
                        row = int(parts[0])  # 提取行号
                        col = int(parts[1])  # 提取列号
                        
                        # 列范围限制为0-3
                        if col <= max_col:
                            if status == 'Train' and 0 <= row <= 2:
                                # 训练集: 行0-2, 列0-3
                                self.selected_patches.append((sample_id, patch_id))
                            elif status == 'Test' and 3 <= row <= 4:
                                # 测试集: 行3-4, 列0-3
                                self.selected_patches.append((sample_id, patch_id))
                    except ValueError:
                        continue  # 跳过无法转换为整数的情况
        # 根据 patch 名称中下划线分割后的第二部分判断 patch 编号
        elif dataset_use == 'Xenium5k_frozenmousebrain':
             for sample_id in sample_name:
            # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
                base_path = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/"
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
                        if status == 'Train' and (train_max_row == -1 or b <= train_max_row):
                            # print('patch_id',patch_id)
                            self.selected_patches.append((sample_id, patch_id))
                        elif status == 'Test' and (train_max_row == -1 or b > train_max_row):
                            self.selected_patches.append((sample_id, patch_id))

                        
                        # # test取全体
                        # if status == 'Train' and (train_max_row == -1 or b <= train_max_row):
                        #     # print('patch_id',patch_id)
                        #     self.selected_patches.append((sample_id, patch_id))
                        # elif status == 'Test':
                        #     self.selected_patches.append((sample_id, patch_id))

                    except ValueError:
                        continue  # 跳过无法转换为整数的情况
                    if status == 'Test' :
                        self.selected_patches.append((sample_id, patch_id))

        elif dataset_use == 'Xenium5k_human':
            for sample_id in sample_name:
                # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
                base_path = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/"
                print('base_path', base_path)
                sub_patches = os.listdir(base_path)
                
                for patch_id in sub_patches:
                    parts = patch_id.split('_')
                    if len(parts) < 2:
                        continue  # 跳过非法格式
                    
                    try:
                        a = int(parts[0])  # 提取行号
                        b = int(parts[1])  # 提取列号
                        
                        # 根据行号划分训练集和测试集
                        # 行号0-4作为训练集，行号5-6作为测试集
                        if status == 'Train' and 0 <= a <= 4:
                            self.selected_patches.append((sample_id, patch_id))
                        elif status == 'Test' and a >= 5:
                            self.selected_patches.append((sample_id, patch_id))
                            
                    except ValueError:
                        continue  # 跳过无法转换为整数的情况
        elif dataset_use == 'Xenium5k_cervixcervicalcancer':
            for sample_id in sample_name:
                # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
                base_path = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/"
                print('base_path', base_path)
                sub_patches = os.listdir(base_path)
                
                for patch_id in sub_patches:
                    parts = patch_id.split('_')
                    if len(parts) < 2:
                        continue  # 跳过非法格式
                    
                    try:
                        a = int(parts[0])  # 提取行号
                        b = int(parts[1])  # 提取列号
                        
                        # 根据行号划分训练集和测试集
                        # 行号0-3作为训练集，行号4+作为测试集
                        if status == 'Train' and 0 <= a <= 4:
                            self.selected_patches.append((sample_id, patch_id))
                        elif status == 'Test' and a >= 5:
                            self.selected_patches.append((sample_id, patch_id))
                            
                    except ValueError:
                        continue  # 跳过无法转换为整数的情况
        elif dataset_use == 'VisiumHD_moueembryo_sorted_data1':
                # 根据 patch 名称中下划线分割后的第二部分判断 patch 编号
                
            for sample_id in sample_name:
                # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
                base_path = f"{data_root}{dataset_use}/HR_ST/"
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


            # 有序输出
            if status == 'Test':
                self.selected_patches.sort(key=lambda x: (int(x[1].split('_')[0]), int(x[1].split('_')[1])))
        
        # 初始化数据容器
        SR_ST_all, spot_ST_all, WSI_5120_all = [], [], []
        self.gene_scale = []
        # print(self.selected_patches)
        # 统一加载逻辑（确保三个数据部分顺序一致）
        for sample_id, patch_id in self.selected_patches:
            # ------------------- 加载HR_ST -------------------
            hr_st_dir = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/{patch_id}/"
            if SR_times == 10:
                SR_ST = np.load(f"{hr_st_dir}HR_ST_256_1000.npz")['arr_0']
            elif SR_times == 5:
                SR_ST = np.load(f"{hr_st_dir}HR_ST_128_1000.npz")['arr_0']
            SR_ST = np.transpose(SR_ST, axes=(2, 0, 1))
            SR_ST_all.append(SR_ST)
            self.gene_scale.append(gene_order)  # 记录基因顺序

            # ------------------- 加载spot_ST -------------------
            spot_st_path = f"{data_root}{dataset_use}/spot_ST/extract/{sample_id}/{patch_id}/spot_ST_1000.npz"
            spot_ST = np.load(spot_st_path)['arr_0']
            spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
            spot_ST_all.append(spot_ST)

            # ------------------- 加载WSI -------------------
            wsi_path = f"{data_root}{dataset_use}/WSI/extract/{sample_id}/{patch_id}/5120_to256.npy"
            WSI_5120 = np.load(wsi_path)
            WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
            WSI_5120_all.append(WSI_5120)

        # 转换为numpy数组
        self.SR_ST_all = np.array(SR_ST_all).astype(np.float64)
        self.SR_ST_all = self.SR_ST_all[:, gene_order, ...].astype(np.float64)
        self.spot_ST_all = np.array(spot_ST_all).astype(np.float64)
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)
        self.WSI_5120_all = np.array(WSI_5120_all)
        self.gene_scale = np.array(self.gene_scale)

        # 数据标准化（保持原有逻辑）
        for data_arr in [self.SR_ST_all, self.spot_ST_all]:
            for ii in range(data_arr.shape[0]):
                for jj in range(data_arr.shape[1]):
                    if np.sum(data_arr[ii, jj]) != 0:
                        Min, Max = data_arr[ii, jj].min(), data_arr[ii, jj].max()
                        data_arr[ii, jj] = (data_arr[ii, jj] - Min) / (Max - Min + 1e-8)
        print('self.SR_ST_all',self.SR_ST_all.shape)
        # print('self.spot_ST_all',self.spot_ST_all.shape)
    def __len__(self):
        return len(self.selected_patches)  # 直接以筛选后的样本数为准

    def __getitem__(self, index):
        # 基因位置编码（保持原有逻辑）
        gene_class = self.gene_scale[index]
        Gene_index_maps = [
            np.ones((256, 256, 1)) * gray_value_of_gene(code, self.gene_order) / 255.0
            for code in gene_class
        ]
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2).transpose(2, 0, 1)
        
        return (
            self.SR_ST_all[index],
            self.spot_ST_all[index],
            self.WSI_5120_all[index],
            final_Gene_index_map
        )


class Xenium_humanbrain(Dataset):
    def __init__(self, data_root, dataset_use,SR_times, status, gene_num, all_gene, gene_order):
        '''
            data_root: 数据根目录路径
            SR_times: 下采样倍数
            status: 数据集状态 'Train' 或 'Test'
            gene_num: 基因数量
            all_gene: 总基因数（用于reshape）
            gene_order: 基因顺序索引
        '''

        if dataset_use == 'Human-brain-glioblastomamultiforme':
            sample_name = ['20240416Brain']
        else:
            raise ValueError('Invalid dataset_use')

        self.selected_patches = []  # 存储符合条件的样本路径
        self.gene_order = gene_order
        # 根据status筛选符合条件的patch
        
        for sample_id in sample_name:
            # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
            base_path = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/"
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

        # 初始化数据容器
        SR_ST_all, spot_ST_all, WSI_5120_all = [], [], []
        self.gene_scale = []

        # 有序输出
        self.selected_patches.sort(key=lambda x: (int(x[1].split('_')[0]), int(x[1].split('_')[1])))
        
        # print(self.selected_patches)
        # 统一加载逻辑（确保三个数据部分顺序一致）
        for sample_id, patch_id in self.selected_patches:
            # ------------------- 加载HR_ST -------------------
            hr_st_dir = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/{patch_id}/"
            if SR_times == 10:
                SR_ST = sp_sparse.load_npz(os.path.join(hr_st_dir, 'HR_ST_256.npz')).toarray().reshape(256, 256, 480)
            elif SR_times == 5:
                SR_ST = sp_sparse.load_npz(os.path.join(hr_st_dir, 'HR_ST_128.npz')).toarray().reshape(128, 128, 480)
            SR_ST = np.transpose(SR_ST, axes=(2, 0, 1))
            SR_ST_all.append(SR_ST)
            self.gene_scale.append(gene_order)  # 记录基因顺序

            # ------------------- 加载spot_ST -------------------
            spot_st_path = f"{data_root}{dataset_use}/spot_ST/extract/{sample_id}/{patch_id}/spot_ST.npz"
            spot_ST = sp_sparse.load_npz(os.path.join(spot_st_path)).toarray().reshape(26, 26, 480)
            spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
            spot_ST_all.append(spot_ST)

            # ------------------- 加载WSI -------------------
            wsi_path = f"{data_root}{dataset_use}/WSI/extract/{sample_id}/{patch_id}/5120_to256.npy"
            WSI_5120 = np.load(wsi_path)
            WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
            WSI_5120_all.append(WSI_5120)

        # 转换为numpy数组
        self.SR_ST_all = np.array(SR_ST_all).astype(np.float64)
        self.SR_ST_all = self.SR_ST_all[:, gene_order, ...].astype(np.float64)
        self.spot_ST_all = np.array(spot_ST_all).astype(np.float64)
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)
        self.WSI_5120_all = np.array(WSI_5120_all)
        self.gene_scale = np.array(self.gene_scale)

        # 数据标准化（保持原有逻辑）
        for data_arr in [self.SR_ST_all, self.spot_ST_all]:
            for ii in range(data_arr.shape[0]):
                for jj in range(data_arr.shape[1]):
                    if np.sum(data_arr[ii, jj]) != 0:
                        Min, Max = data_arr[ii, jj].min(), data_arr[ii, jj].max()
                        data_arr[ii, jj] = (data_arr[ii, jj] - Min) / (Max - Min + 1e-8)
        print('self.SR_ST_all',self.SR_ST_all.shape)
        print('self.spot_ST_all',self.spot_ST_all.shape)
    def __len__(self):
        return len(self.selected_patches)  # 直接以筛选后的样本数为准

    def __getitem__(self, index):
        # 基因位置编码（保持原有逻辑）
        gene_class = self.gene_scale[index]
        Gene_index_maps = [
            np.ones((256, 256, 1)) * gray_value_of_gene(code, self.gene_order) / 255.0
            for code in gene_class
        ]
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2).transpose(2, 0, 1)
        
        return (
            self.SR_ST_all[index],
            self.spot_ST_all[index],
            self.WSI_5120_all[index],
            final_Gene_index_map
        )
    
# class Xenium_dataset(Dataset):
#     def __init__(self, data_root, SR_times, status, gene_num,all_gene):
#         '''
#             data_root: 数据根目录的路径。
#             SR_times: 下采样倍数，影响加载的HR ST数据的分辨率。
#             status: 指定数据集的状态，值为 'Train' 或 'Test'，用于选择不同的样本。
#             gene_num: 需要处理的基因数量。
#         '''
#         if status == 'Train':
#             sample_name = ['01220101', '01220102', 'NC1', 'NC2']#, '0418'

#         elif status == 'Test':
#             sample_name = ['01220201', '01220202']

#         self.all_gene = all_gene
#         gene_order_path = os.path.join(data_root, 'gene_order1.npy')
#         gene_order = np.load(gene_order_path)[0:all_gene]
#         # print(gene_order)
#         self.gene_order = gene_order
#         SR_ST_all = []
#         self.gene_scale = []#new1.5:
#         self.gene_num = gene_num#new1.5:
        
#         # 
#         # 
#         ### HR ST
#         for sample_id in sample_name:
#             sub_patches=os.listdir(data_root+'Xenium/HR_ST/extract/'+sample_id)
#             for patch_id in sub_patches:
#                 if SR_times==10:
#                     print(data_root+'Xenium/HR_ST/extract/'+sample_id+'/'+patch_id+'/HR_ST_256.npz')
#                     SR_ST=sp_sparse.load_npz(data_root+'Xenium/HR_ST/extract/'+sample_id+'/'+patch_id+'/HR_ST_256.npz').toarray().reshape(256, 256, 280)
#                 elif SR_times==5:
#                     SR_ST = sp_sparse.load_npz(data_root+'Xenium/HR_ST/extract/'+sample_id+'/'+patch_id+'/HR_ST_128.npz').toarray().reshape(128, 128, 280)
#                 SR_ST=np.transpose(SR_ST,axes=(2,0,1))
#                 SR_ST_all.append(SR_ST)
#                 self.gene_scale.append(gene_order)#new1.5:
#         self.SR_ST_all = np.array(SR_ST_all)
#         #new1.5:
#         self.gene_scale = np.array(self.gene_scale)
#         # self.gene_scale = np.reshape(self.gene_scale, (self.gene_scale.shape[0]*(all_gene//gene_num),gene_num))
#         #new1.5:
#         # 新
#         self.SR_ST_all = self.SR_ST_all[:, gene_order, ...].astype(np.float64)

#         # print('SR_ST_all',self.SR_ST_all.shape)
#         # self.SR_ST_all = np.reshape(self.SR_ST_all, (self.SR_ST_all.shape[0]*(all_gene//gene_num),gene_num, self.SR_ST_all.shape[2],self.SR_ST_all.shape[3]))
#         # print('SR_ST_all',self.SR_ST_all.shape)
#         for ii in range(self.SR_ST_all.shape[0]):  # 初始化SR_ST_all列表，用于存储加载的HR ST数据。
#             for jj in range(self.SR_ST_all.shape[1]):
#                 if np.sum(self.SR_ST_all[ii, jj])!= 0:
#                     Max = np.max(self.SR_ST_all[ii, jj])
#                     Min = np.min(self.SR_ST_all[ii, jj])
#                     self.SR_ST_all[ii, jj] = (self.SR_ST_all[ii, jj] - Min) / (Max - Min)  # 对选择的基因进行归一化处理

#         ### spot ST
#         spot_ST_all = []
#         for sample_id in sample_name:
#             sub_patches = os.listdir(data_root + 'Xenium/spot_ST/extract/' + sample_id)
#             for patch_id in sub_patches:
#                 spot_ST = np.load(data_root + 'Xenium/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
#                 spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
#                 spot_ST_all.append(spot_ST)
#         self.spot_ST_all = np.array(spot_ST_all)

#         # 新
#         self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)

#         # print('spot_ST_all',self.spot_ST_all.shape)
#         # self.spot_ST_all = np.reshape(self.spot_ST_all, (self.spot_ST_all.shape[0]*(all_gene//gene_num),gene_num, self.spot_ST_all.shape[2],self.spot_ST_all.shape[3]))
#         # print('spot_ST_all',self.spot_ST_all.shape)
#         for ii in range(self.spot_ST_all.shape[0]):
#             for jj in range(self.spot_ST_all.shape[1]):
#                 if np.sum(self.spot_ST_all[ii, jj])!= 0:
#                     Max = np.max(self.spot_ST_all[ii, jj])
#                     Min = np.min(self.spot_ST_all[ii, jj])
#                     self.spot_ST_all[ii, jj] = (self.spot_ST_all[ii, jj] - Min) / (Max - Min)

#         ### WSI 5120
#         WSI_5120_all = []
#         for sample_id in sample_name:
#             sub_patches = os.listdir(data_root + 'Xenium/WSI/extract/' + sample_id)
#             for patch_id in sub_patches:
#                 WSI_5120 = np.load(data_root + 'Xenium/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
#                 max = np.max(WSI_5120)
#                 WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
#                 WSI_5120_all.append(WSI_5120)
#         self.WSI_5120_all = np.array(WSI_5120_all)
#              # 对WSI 5120数据按照分组数量复制到batch维度
#         # self.WSI_5120_all = np.repeat(self.WSI_5120_all, all_gene//gene_num, axis = 0)

#     def __len__(self):
#         # 返回处理后数据的batch维度大小（也就是分组后的数量）
#         return self.SR_ST_all.shape[0]

#     def __getitem__(self, index):
#         '''
#             返回对应索引位置处理后的各项数据，包含分组融入batch维度后的ST数据以及对应复制后的WSI数据。
#         ''' 
#         #new1.5:
#         gene_class = self.gene_scale[index]

#         Gene_index_maps = []

#         for gene_code in gene_class:
#             Gene_codes = gray_value_of_gene(gene_code, self.gene_order)
#             Gene_index_map = np.ones(shape=(256, 256,1)) * Gene_codes / 255.0
#             Gene_index_maps.append(Gene_index_map)
#         final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2)
#         final_Gene_index_map = np.moveaxis(final_Gene_index_map, 2, 0)
#         #new1.5:

#         return self.SR_ST_all[index], self.spot_ST_all[index], self.WSI_5120_all[index], final_Gene_index_map #new1.5:, self.WSI_320_all[index],

class Xenium5k_dataset(Dataset):
    def __init__(self, data_root, dataset_use, SR_times, status, gene_num, all_gene, gene_order, gene_name_order):
        # 固定使用一个样本ID，并按 patch 划分训练与测试
        if dataset_use == 'Xenium5k_frozenmousebrain':
            sample_name = ['20240801mousebrain']
        elif dataset_use == 'Xenium5k_ovaryovariancancer':
            sample_name = ['20241218ovary']
        elif dataset_use == 'Xenium5k_cervixcervicalcancer':
            sample_name = ['20240905cervix']
        elif dataset_use == 'Xenium5k_human':
            sample_name = ['20241024Breast']
        elif dataset_use == 'Xenium5k_brain-glioblastomamultiforme':
            sample_name = ['20240416Brain']
        else:
            raise ValueError('Invalid dataset_use')
            
        self.selected_patches = []  # 存储符合条件的 (sample_id, patch_id)
        self.gene_order = gene_order
        self.gene_name_order = gene_name_order
        self.gene_num = gene_num
        
        if dataset_use == 'Xenium5k_ovaryovariancancer':
            max_col = 3  # 列范围 0-3
            
            for sample_id in sample_name:
                base_path = f"{data_root}/HR_ST/extract/{sample_id}/"
                print('base_path', base_path)
                sub_patches = os.listdir(base_path)
                
                for patch_id in sub_patches:
                    parts = patch_id.split('_')
                    if len(parts) < 2:
                        continue  # 跳过非法格式
                        
                    try:
                        row = int(parts[0])  # 提取行号
                        col = int(parts[1])  # 提取列号
                        
                        # 列范围限制为0-3
                        if col <= max_col:
                            if status == 'Train' and 0 <= row <= 2:
                                # 训练集: 行0-2, 列0-3
                                self.selected_patches.append((sample_id, patch_id))
                            elif status == 'Test' and 3 <= row <= 4:
                                # 测试集: 行3-4, 列0-3
                                self.selected_patches.append((sample_id, patch_id))
                    except ValueError:
                        continue  # 跳过无法转换为整数的情况
        # 根据 patch 名称中下划线分割后的第二部分判断 patch 编号
        elif dataset_use == 'Xenium5k_frozenmousebrain':
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
                        if status == 'Train' and (train_max_row == -1 or b <= train_max_row):
                            # print('patch_id',patch_id)
                            self.selected_patches.append((sample_id, patch_id))
                        elif status == 'Test' and (train_max_row == -1 or b > train_max_row):
                            self.selected_patches.append((sample_id, patch_id))
                        
                        # # test取全体
                        # if status == 'Train' and (train_max_row == -1 or b <= train_max_row):
                        #     # print('patch_id',patch_id)
                        #     self.selected_patches.append((sample_id, patch_id))
                        # elif status == 'Test':
                        #     self.selected_patches.append((sample_id, patch_id))

                    except ValueError:
                        continue  # 跳过无法转换为整数的情况

        elif dataset_use == 'Xenium5k_human':
            for sample_id in sample_name:
                # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
                base_path = f"{data_root}/HR_ST/extract/{sample_id}/"
                print('base_path', base_path)
                sub_patches = os.listdir(base_path)
                
                for patch_id in sub_patches:
                    parts = patch_id.split('_')
                    if len(parts) < 2:
                        continue  # 跳过非法格式
                    
                    try:
                        a = int(parts[0])  # 提取行号
                        b = int(parts[1])  # 提取列号
                        
                        # 根据行号划分训练集和测试集
                        # 行号0-4作为训练集，行号5-6作为测试集
                        if status == 'Train' and 0 <= a <= 4:
                            self.selected_patches.append((sample_id, patch_id))
                        elif status == 'Test' and a >= 5:
                            self.selected_patches.append((sample_id, patch_id))
                            
                    except ValueError:
                        continue  # 跳过无法转换为整数的情况
        elif dataset_use == 'Xenium5k_cervixcervicalcancer':
            for sample_id in sample_name:
                # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
                base_path = f"{data_root}/HR_ST/extract/{sample_id}/"
                print('base_path', base_path)
                sub_patches = os.listdir(base_path)
                
                for patch_id in sub_patches:
                    parts = patch_id.split('_')
                    if len(parts) < 2:
                        continue  # 跳过非法格式
                    
                    try:
                        a = int(parts[0])  # 提取行号
                        b = int(parts[1])  # 提取列号
                        
                        # 根据行号划分训练集和测试集
                        # 行号0-3作为训练集，行号4+作为测试集
                        if status == 'Train' and 0 <= a <= 4:
                            self.selected_patches.append((sample_id, patch_id))
                        elif status == 'Test' and a >= 5:
                            self.selected_patches.append((sample_id, patch_id))
                            
                    except ValueError:
                        continue  # 跳过无法转换为整数的情况
                    
        # 加载 HR ST 数据
        SR_ST_all = []
        self.gene_scale = []
        for sample_id, patch_id in self.selected_patches:
            hr_st_dir = os.path.join(data_root,  "HR_ST", "extract", sample_id, patch_id)
            if SR_times == 10:
                SR_ST = sp_sparse.load_npz(os.path.join(data_root,'HR_ST/extract', sample_id, patch_id, 'HR_ST_256.npz')).toarray().reshape(256, 256, 10414)
                SR_ST = np.transpose(SR_ST, axes=(2, 0, 1))  # 转为 (channels, H, W)
            elif SR_times == 5:
                SR_ST = sp_sparse.load_npz(os.path.join(data_root,'HR_ST/extract', sample_id, patch_id, 'HR_ST_128.npz')).toarray().reshape(128, 128, 10414)
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
            spot_ST = sp_sparse.load_npz(os.path.join(data_root, 'spot_ST/extract', sample_id, patch_id, 'spot_ST.npz')).toarray().reshape(26, 26, 10414)
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
        self.tokenizer = AutoTokenizer.from_pretrained("/home/zeiler/NMI/bert",
                                                         local_files_only=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained("/home/zeiler/NMI/bert",
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



class Xenium5k2(Dataset):
    def __init__(self, data_root, dataset_use,SR_times, status, gene_num, all_gene, gene_order,gene_name_order):
        '''
            data_root: 数据根目录路径
            SR_times: 下采样倍数
            status: 数据集状态 'Train' 或 'Test'
            gene_num: 基因数量
            all_gene: 总基因数（用于reshape）
            gene_order: 基因顺序索引
        '''

        if dataset_use == 'VisiumHD_mouseembryo_sorted_data1':
            sample_name = ['20240611mouseembryo']
        else:
            raise ValueError('Invalid dataset_use')

        self.selected_patches = []  # 存储符合条件的样本路径
        self.gene_order = gene_order
        self.gene_name_order = gene_name_order
        self.gene_num = gene_num
        # 根据status筛选符合条件的patch

        if dataset_use == 'VisiumHD_mouseembryo_sorted_data1':
                # 根据 patch 名称中下划线分割后的第二部分判断 patch 编号
                
            for sample_id in sample_name:
                # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
                base_path = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}"
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
                        elif status == 'Test' and (train_max_row == -1 or b > train_max_row):
                            self.selected_patches.append((sample_id, patch_id))

                    except ValueError:
                        continue  # 跳过无法转换为整数的情况
            # 初始化数据容器
            SR_ST_all, spot_ST_all, WSI_5120_all,WSI_320_all = [], [], [],[]
            self.gene_scale = []

            # 有序输出
            if status == 'Test':
                self.selected_patches.sort(key=lambda x: (int(x[1].split('_')[0]), int(x[1].split('_')[1])))
        #print('self.selected_patches',self.selected_patches)
        # print(self.selected_patches)
        # 统一加载逻辑（确保三个数据部分顺序一致）
        for sample_id, patch_id in self.selected_patches:
            # ------------------- 加载HR_ST -------------------
            hr_st_dir = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/{patch_id}/"
            if SR_times == 10:
                #print(os.path.join(data_root,dataset_use,'HR_ST/extract/', patch_id, 'HR_ST_256.npz'))
                SR_ST = sp_sparse.load_npz(os.path.join(data_root,dataset_use,'HR_ST/extract/',sample_id, patch_id, 'HR_ST_256.npz')).toarray().reshape(256, 256, 10417)
            elif SR_times == 5:
                SR_ST = sp_sparse.load_npz(os.path.join(data_root,dataset_use,'HR_ST/extract/',sample_id, patch_id, 'HR_ST_128.npz')).toarray().reshape(128, 128, 10417)
            SR_ST = np.transpose(SR_ST, axes=(2, 0, 1))
            SR_ST_all.append(SR_ST)
            self.gene_scale.append(gene_order)  # 记录基因顺序

            # ------------------- 加载spot_ST -------------------
            spot_st_path = f"{data_root}{dataset_use}/spot_ST/{patch_id}/spot_ST_1000.npz"
            spot_ST = sp_sparse.load_npz(os.path.join(data_root,dataset_use, 'spot_ST/extract/',sample_id, patch_id, 'spot_ST.npz')).toarray().reshape(26, 26, 10417)
            spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
            spot_ST_all.append(spot_ST)

            # ------------------- 加载WSI -------------------
            wsi_path = f"{data_root}{dataset_use}/WSI/extract/20240611mouseembryo/{patch_id}/5120_to256.npy"
            WSI_5120 = np.load(wsi_path)
            WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
            WSI_5120_all.append(WSI_5120)
            wsi_320_path = os.path.join(data_root,dataset_use, "WSI", "extract", sample_id, patch_id, "320_to16.npy")
            #print(wsi_320_path)
            WSI_320 = np.load(wsi_320_path)
            WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
            WSI_320_all.append(WSI_320)
       
        # 转换为numpy数组
        self.SR_ST_all = np.array(SR_ST_all).astype(np.float64)
        #print(gene_order)
        #print(self.SR_ST_all.shape)
        self.SR_ST_all = self.SR_ST_all[:, gene_order, ...].astype(np.float64)
        self.spot_ST_all = np.array(spot_ST_all).astype(np.float64)
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)
        self.WSI_5120_all = np.array(WSI_5120_all)
        self.WSI_320_all = np.array(WSI_320_all)

        self.gene_scale = np.array(self.gene_scale)

        # 数据标准化（保持原有逻辑）
        for data_arr in [self.SR_ST_all, self.spot_ST_all]:
            for ii in range(data_arr.shape[0]):
                for jj in range(data_arr.shape[1]):
                    if np.sum(data_arr[ii, jj]) != 0:
                        Min, Max = data_arr[ii, jj].min(), data_arr[ii, jj].max()
                        data_arr[ii, jj] = (data_arr[ii, jj] - Min) / (Max - Min + 1e-8)
        #print('self.SR_ST_all',self.SR_ST_all.shape)
        # print('self.spot_ST_all',self.spot_ST_all.shape)
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
        print(f"Test模式下，选中的patch数量：{len(self.selected_patches)}")
    def __len__(self):
        return len(self.selected_patches)  # 直接以筛选后的样本数为准

    def __getitem__(self, index):
        # 基因位置编码（保持原有逻辑）
        gene_class = self.gene_scale[index]
        Gene_index_maps = [
            np.ones((256, 256, 1)) * gray_value_of_gene(code, self.gene_order) / 255.0
            for code in gene_class
        ]
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2).transpose(2, 0, 1)
        
        return (
            self.SR_ST_all[index],
            self.spot_ST_all[index],
            self.WSI_5120_all[index],
            self.WSI_320_all[index],
            gene_class,
            final_Gene_index_map,
            self.gene_name_features,   # shape: [gene_num, hidden_size]
            self.metadata_feature 
        )


class Visiumhdmousekidney(Dataset):
    def __init__(self, data_root, dataset_use,SR_times, status, gene_num, all_gene, gene_order,gene_name_order):
        '''
            data_root: 数据根目录路径
            SR_times: 下采样倍数
            status: 数据集状态 'Train' 或 'Test'
            gene_num: 基因数量
            all_gene: 总基因数（用于reshape）
            gene_order: 基因顺序索引
        '''

        if dataset_use == 'Visiumhd_mouse_kidney':
            sample_name = ['20240611mousekidney']
        else:
            raise ValueError('Invalid dataset_use')

        self.selected_patches = []  # 存储符合条件的样本路径
        self.gene_order = gene_order
        self.gene_name_order = gene_name_order
        self.gene_num = gene_num
        # 根据status筛选符合条件的patch

        if dataset_use == 'Visiumhd_mouse_kidney':
                # 根据 patch 名称中下划线分割后的第二部分判断 patch 编号
                
            for sample_id in sample_name:
                # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
                base_path = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}"
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
                        elif status == 'Test' and (train_max_row == -1 or b > train_max_row):
                            self.selected_patches.append((sample_id, patch_id))

                    except ValueError:
                        continue  # 跳过无法转换为整数的情况
            # 初始化数据容器
            SR_ST_all, spot_ST_all, WSI_5120_all,WSI_320_all = [], [], [],[]
            self.gene_scale = []

            # 有序输出
            if status == 'Test':
                self.selected_patches.sort(key=lambda x: (int(x[1].split('_')[0]), int(x[1].split('_')[1])))
        #print('self.selected_patches',self.selected_patches)
        # print(self.selected_patches)
        # 统一加载逻辑（确保三个数据部分顺序一致）
        for sample_id, patch_id in self.selected_patches:
            # ------------------- 加载HR_ST -------------------
            hr_st_dir = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/{patch_id}/"
            if SR_times == 10:
                #print(os.path.join(data_root,dataset_use,'HR_ST/extract/', patch_id, 'HR_ST_256.npz'))
                SR_ST = sp_sparse.load_npz(os.path.join(data_root,dataset_use,'HR_ST/extract/',sample_id, patch_id, 'HR_ST_256.npz')).toarray().reshape(256, 256, 200)
            elif SR_times == 5:
                SR_ST = sp_sparse.load_npz(os.path.join(data_root,dataset_use,'HR_ST/extract/',sample_id, patch_id, 'HR_ST_128.npz')).toarray().reshape(128, 128, 200)
            SR_ST = np.transpose(SR_ST, axes=(2, 0, 1))
            SR_ST_all.append(SR_ST)
            self.gene_scale.append(gene_order)  # 记录基因顺序

            # ------------------- 加载spot_ST -------------------
            spot_st_path = f"{data_root}{dataset_use}/spot_ST/{patch_id}/spot_ST_1000.npz"
            spot_ST = sp_sparse.load_npz(os.path.join(data_root,dataset_use, 'spot_ST/extract/',sample_id, patch_id, 'spot_ST.npz')).toarray().reshape(26, 26, 200)
            spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
            spot_ST_all.append(spot_ST)

            # ------------------- 加载WSI -------------------
            wsi_path = f"{data_root}{dataset_use}/WSI/extract/20240611mousekidney/{patch_id}/5120_to256.npy"
            WSI_5120 = np.load(wsi_path)
            WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
            WSI_5120_all.append(WSI_5120)
            wsi_320_path = os.path.join(data_root,dataset_use, "WSI", "extract", sample_id, patch_id, "320_to16.npy")
            #print(wsi_320_path)
            WSI_320 = np.load(wsi_320_path)
            WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
            WSI_320_all.append(WSI_320)
       
        # 转换为numpy数组
        self.SR_ST_all = np.array(SR_ST_all).astype(np.float64)
        #print(gene_order)
        #print(self.SR_ST_all.shape)
        self.SR_ST_all = self.SR_ST_all[:, gene_order, ...].astype(np.float64)
        self.spot_ST_all = np.array(spot_ST_all).astype(np.float64)
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)
        self.WSI_5120_all = np.array(WSI_5120_all)
        self.WSI_320_all = np.array(WSI_320_all)

        self.gene_scale = np.array(self.gene_scale)

        # 数据标准化（保持原有逻辑）
        for data_arr in [self.SR_ST_all, self.spot_ST_all]:
            for ii in range(data_arr.shape[0]):
                for jj in range(data_arr.shape[1]):
                    if np.sum(data_arr[ii, jj]) != 0:
                        Min, Max = data_arr[ii, jj].min(), data_arr[ii, jj].max()
                        data_arr[ii, jj] = (data_arr[ii, jj] - Min) / (Max - Min + 1e-8)
        #print('self.SR_ST_all',self.SR_ST_all.shape)
        # print('self.spot_ST_all',self.spot_ST_all.shape)
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
        
        metadata_prompt = ("Provide spatial transcriptomics data from the VisiumHD platform for mouse species, with a healthy condition, and kidney tissue type.")
        with torch.no_grad():
            inputs = self.tokenizer(metadata_prompt, return_tensors="pt")
            outputs = self.model(**inputs)
            self.metadata_feature = outputs.pooler_output.squeeze(0)
        print('metadata_feature shape:', self.metadata_feature.shape)
        print(f"Test模式下，选中的patch数量：{len(self.selected_patches)}")
    def __len__(self):
        return len(self.selected_patches)  # 直接以筛选后的样本数为准

    def __getitem__(self, index):
        # 基因位置编码（保持原有逻辑）
        gene_class = self.gene_scale[index]
        Gene_index_maps = [
            np.ones((256, 256, 1)) * gray_value_of_gene(code, self.gene_order) / 255.0
            for code in gene_class
        ]
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2).transpose(2, 0, 1)
        
        return (
            self.SR_ST_all[index],
            self.spot_ST_all[index],
            self.WSI_5120_all[index],
            self.WSI_320_all[index],
            gene_class,
            final_Gene_index_map,
            self.gene_name_features,   # shape: [gene_num, hidden_size]
            self.metadata_feature 
        )
        


class VisiumhdhumanTonsil(Dataset):
    def __init__(self, data_root, dataset_use,SR_times, status, gene_num, all_gene, gene_order,gene_name_order):
        '''
            data_root: 数据根目录路径
            SR_times: 下采样倍数
            status: 数据集状态 'Train' 或 'Test'
            gene_num: 基因数量
            all_gene: 总基因数（用于reshape）
            gene_order: 基因顺序索引
        '''

        if dataset_use == 'Visiumhd_human_Tonsil':
            sample_name = ['20250220humantonsil']
        else:
            raise ValueError('Invalid dataset_use')

        self.selected_patches = []  # 存储符合条件的样本路径
        self.gene_order = gene_order
        self.gene_name_order = gene_name_order
        self.gene_num = gene_num
        # 根据status筛选符合条件的patch

        if dataset_use == 'Visiumhd_human_Tonsil':
                # 根据 patch 名称中下划线分割后的第二部分判断 patch 编号
                
            for sample_id in sample_name:
                # 以HR_ST目录为基准获取所有patch列表（假设其他数据目录结构相同）
                base_path = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}"
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
                        elif status == 'Test' and (train_max_row == -1 or b > train_max_row):
                            self.selected_patches.append((sample_id, patch_id))

                    except ValueError:
                        continue  # 跳过无法转换为整数的情况
            # 初始化数据容器
            SR_ST_all, spot_ST_all, WSI_5120_all,WSI_320_all = [], [], [],[]
            self.gene_scale = []

            # 有序输出
            if status == 'Test':
                self.selected_patches.sort(key=lambda x: (int(x[1].split('_')[0]), int(x[1].split('_')[1])))
        #print('self.selected_patches',self.selected_patches)
        # print(self.selected_patches)
        # 统一加载逻辑（确保三个数据部分顺序一致）
        for sample_id, patch_id in self.selected_patches:
            # ------------------- 加载HR_ST -------------------
            hr_st_dir = f"{data_root}{dataset_use}/HR_ST/extract/{sample_id}/{patch_id}/"
            if SR_times == 10:
                #print(os.path.join(data_root,dataset_use,'HR_ST/extract/', patch_id, 'HR_ST_256.npz'))
                SR_ST = sp_sparse.load_npz(os.path.join(data_root,dataset_use,'HR_ST/extract/',sample_id, patch_id, 'HR_ST_256.npz')).toarray().reshape(256, 256, 200)
            elif SR_times == 5:
                SR_ST = sp_sparse.load_npz(os.path.join(data_root,dataset_use,'HR_ST/extract/',sample_id, patch_id, 'HR_ST_128.npz')).toarray().reshape(128, 128, 200)
            SR_ST = np.transpose(SR_ST, axes=(2, 0, 1))
            SR_ST_all.append(SR_ST)
            self.gene_scale.append(gene_order)  # 记录基因顺序

            # ------------------- 加载spot_ST -------------------
            spot_st_path = f"{data_root}{dataset_use}/spot_ST/{patch_id}/spot_ST_1000.npz"
            spot_ST = sp_sparse.load_npz(os.path.join(data_root,dataset_use, 'spot_ST/extract/',sample_id, patch_id, 'spot_ST.npz')).toarray().reshape(26, 26, 200)
            spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
            spot_ST_all.append(spot_ST)

            # ------------------- 加载WSI -------------------
            wsi_path = f"{data_root}{dataset_use}/WSI/extract/20240611mousekidney/{patch_id}/5120_to256.npy"
            WSI_5120 = np.load(wsi_path)
            WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
            WSI_5120_all.append(WSI_5120)
            wsi_320_path = os.path.join(data_root,dataset_use, "WSI", "extract", sample_id, patch_id, "320_to16.npy")
            #print(wsi_320_path)
            WSI_320 = np.load(wsi_320_path)
            WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
            WSI_320_all.append(WSI_320)
       
        # 转换为numpy数组
        self.SR_ST_all = np.array(SR_ST_all).astype(np.float64)
        #print(gene_order)
        #print(self.SR_ST_all.shape)
        self.SR_ST_all = self.SR_ST_all[:, gene_order, ...].astype(np.float64)
        self.spot_ST_all = np.array(spot_ST_all).astype(np.float64)
        self.spot_ST_all = self.spot_ST_all[:, gene_order, ...].astype(np.float64)
        self.WSI_5120_all = np.array(WSI_5120_all)
        self.WSI_320_all = np.array(WSI_320_all)

        self.gene_scale = np.array(self.gene_scale)

        # 数据标准化（保持原有逻辑）
        for data_arr in [self.SR_ST_all, self.spot_ST_all]:
            for ii in range(data_arr.shape[0]):
                for jj in range(data_arr.shape[1]):
                    if np.sum(data_arr[ii, jj]) != 0:
                        Min, Max = data_arr[ii, jj].min(), data_arr[ii, jj].max()
                        data_arr[ii, jj] = (data_arr[ii, jj] - Min) / (Max - Min + 1e-8)
        #print('self.SR_ST_all',self.SR_ST_all.shape)
        # print('self.spot_ST_all',self.spot_ST_all.shape)
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
        
        metadata_prompt = ("Provide spatial transcriptomics data from the VisiumHD platform for human species, with a healthy condition, and Tonsil tissue type.")
        with torch.no_grad():
            inputs = self.tokenizer(metadata_prompt, return_tensors="pt")
            outputs = self.model(**inputs)
            self.metadata_feature = outputs.pooler_output.squeeze(0)
        print('metadata_feature shape:', self.metadata_feature.shape)
        print(f"Test模式下，选中的patch数量：{len(self.selected_patches)}")
    def __len__(self):
        return len(self.selected_patches)  # 直接以筛选后的样本数为准

    def __getitem__(self, index):
        # 基因位置编码（保持原有逻辑）
        gene_class = self.gene_scale[index]
        Gene_index_maps = [
            np.ones((256, 256, 1)) * gray_value_of_gene(code, self.gene_order) / 255.0
            for code in gene_class
        ]
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2).transpose(2, 0, 1)
        
        return (
            self.SR_ST_all[index],
            self.spot_ST_all[index],
            self.WSI_5120_all[index],
            self.WSI_320_all[index],
            gene_class,
            final_Gene_index_map,
            self.gene_name_features,   # shape: [gene_num, hidden_size]
            self.metadata_feature 
        )