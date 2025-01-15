import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn

# 输出torch版本号
# print(torch.__version__)
# 设置pytorch中默认的浮点类型
torch.set_default_dtype(torch.float32)  # 使用torch.float32替代torch.FloatTensor

# 使用pandas读取两个文件
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')


# 训练数据集包括1460个样本、80个特征和1个标签
# print(train_data.shape)
# 测试数据集包括1459个样本和80个特征，需要将测试数据集中每个样本的标签预测出来
# print(test_data.shape)

# [:,1:-1]所有行的第一列到-2列，[:,1:]
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 打印所有特征的数量 (2919,79)
# print(all_features.shape)

# object是str类或数字混合类型（mixed），将特征为数字的列单独拿出。并保存列名到numeric
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index  # 排除object类型的特征

# 打印非离散的数据类型
# print(numeric_features)

# 对每种数字型的特征(每一的数字)进行标准化
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

all_features[numeric_features] = all_features[numeric_features].fillna(0)

# dummpy_na=True将缺失值也当做合法的特征值并为其创建指示特征
# get_dummies即上述将离散值转换为指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)

#print(all_features.shape)

# 取train_data的行数，即训练集个数
n_train = train_data.shape[0] 

# 训练集特征
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
# 测试集特征
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
# 训练集标签
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

## 模型训练
loss = torch.nn.MSELoss()

def get_net(feature_num):
	# 实例化nn
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net
