import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns

# 加载数据
gt_arr = np.load('/date/zjz/job_2/data/samples/Channel_npy_files/3_1_gt_all_genes.npy')
pred_arr = np.load('/date/zjz/job_2/data/samples/Channel_npy_files/3_1_pred_all_genes.npy')

print(gt_arr.shape)
print(pred_arr.shape)

