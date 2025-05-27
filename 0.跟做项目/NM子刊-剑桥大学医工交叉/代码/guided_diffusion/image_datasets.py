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

def load_data(data_root,dataset_use,status,SR_times,gene_num,all_gene,gene_order=None
):

    if 'Xenium5k_' in dataset_use:
       dataset= Xenium5k(data_root,dataset_use,SR_times,status,gene_num,all_gene,gene_order)
    elif dataset_use=='Human-brain-glioblastomamultiforme':
        dataset=Xenium_humanbrain(data_root,dataset_use,SR_times,status,gene_num,all_gene,gene_order)
    return dataset
class Spatialinformation(Dataset):
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
                        # elif status == 'Test' and (train_max_row == -1 or b > train_max_row):
                        #     self.selected_patches.append((sample_id, patch_id))
                        if status == 'Test' :
                            self.selected_patches.append((sample_id, patch_id))
                    except ValueError:
                        continue  # 跳过无法转换为整数的情况


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
        # return self.selected_patches
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
            print('TEST Order1:',self.selected_patches)
        
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
            np.ones((64, 64, 1)) * gray_value_of_gene(code, self.gene_order) / 255.0
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
            np.ones((64, 64, 1)) * gray_value_of_gene(code, self.gene_order) / 255.0
            for code in gene_class
        ]
        final_Gene_index_map = np.concatenate(Gene_index_maps, axis=2).transpose(2, 0, 1)
        
        return (
            self.SR_ST_all[index],
            self.spot_ST_all[index],
            self.WSI_5120_all[index],
            final_Gene_index_map
        )
    
