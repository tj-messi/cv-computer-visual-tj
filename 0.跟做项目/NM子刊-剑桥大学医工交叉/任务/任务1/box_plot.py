import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns

# 加载数据
gt_arr = np.load('/date/zjz/job_1/data/concatenated_gt_all_genes.npy')
pred_arr = np.load('/date/zjz/job_1/data/concatenated_pred_all_genes.npy')


def calculate_lwhm(intensity_values):
    peak_intensity = np.max(intensity_values)
    peak_index = np.argmax(intensity_values)
    
    half_max_intensity = peak_intensity / 2
 
    left_index = peak_index
    while left_index > 0 and intensity_values[left_index] > half_max_intensity:
        left_index -= 1
    
    lwhm = peak_index - left_index

    return lwhm,peak_index,left_index


lwhm_gt_list = []
lwhm_pred_list = []

for i in range(0,1000):
    channel_gt_arr = gt_arr[:, :, i]  # 形状 (384, 640)
    channel_pred_arr = pred_arr[:, :, i]

    # 列索引平均
    channel_gt_avg = np.mean(channel_gt_arr, axis=1)  
    channel_pred_avg = np.mean(channel_pred_arr, axis=1) 

    lwhm_gt,peak_index_gt,left_index_gt = calculate_lwhm(channel_gt_avg)
    lwhm_pred,peak_index_pred,left_index_pred = calculate_lwhm(channel_pred_avg)

    lwhm_gt_list.append(lwhm_gt)
    lwhm_pred_list.append(lwhm_pred)

# print(len(lwhm_gt_list)) 1000
# Create a box plot for LWHM values of ground truth and prediction
plt.figure(figsize=(10, 6))
plt.boxplot([lwhm_gt_list, lwhm_pred_list], labels=['Ground Truth', 'Prediction'])

# Set plot title and labels
plt.title("LWHM Comparison: Ground Truth vs Prediction")
plt.ylabel("LWHM")

# Show the plot
plt.show()

# Save the box plot as an image
plt.savefig('/date/zjz/job_1/lwhm_boxplot.png')
plt.close()