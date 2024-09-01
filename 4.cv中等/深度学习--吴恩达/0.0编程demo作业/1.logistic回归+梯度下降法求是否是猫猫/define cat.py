import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# 加载数据集
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 预处理图片的几个数据：训练数量、测试数量、图片高度和宽度
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# 转换numpy数组为展开的(num_px * num_px * 3, 1)的矩阵
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T


# 归一化数据
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.



