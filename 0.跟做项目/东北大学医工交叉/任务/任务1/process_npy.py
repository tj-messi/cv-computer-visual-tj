import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns

# 加载数据
gt_arr = np.load('/date/zjz/job_1/data/concatenated_gt_all_genes.npy')
pred_arr = np.load('/date/zjz/job_1/data/concatenated_pred_all_genes.npy')

# 取第一个基因
first_channel_gt_arr = gt_arr[:, :, 0]  # 形状 (384, 640)
first_channel_pred_arr = pred_arr[:, :, 0]

# 列索引平均
first_channel_gt_avg = np.mean(first_channel_gt_arr, axis=1)  
first_channel_pred_avg = np.mean(first_channel_pred_arr, axis=1)  

def calculate_lwhm(intensity_values):
    peak_intensity = np.max(intensity_values)
    peak_index = np.argmax(intensity_values)
    
    half_max_intensity = peak_intensity / 2
 
    left_index = peak_index
    while left_index >= 0 and intensity_values[left_index] > half_max_intensity:
        left_index -= 1
    
    lwhm = peak_index - left_index

    return lwhm,peak_index,left_index

lwhm_gt,peak_index_gt,left_index_gt = calculate_lwhm(first_channel_gt_avg)
lwhm_pred,peak_index_pred,left_index_pred = calculate_lwhm(first_channel_pred_avg)

print(left_index_gt , left_index_pred)

import matplotlib.pyplot as plt

# Plot the intensity distributions and LWHM markers for ground truth and prediction
plt.figure(figsize=(10, 6))

# Plot the ground truth and prediction
plt.plot(first_channel_gt_avg, label='Ground Truth', color='blue')
plt.plot(first_channel_pred_avg, label='Prediction', color='orange')

# Plot the LWHM boundaries for ground truth (left and peak indices)
plt.axvline(x=left_index_gt, color='blue', linestyle='--', label='Left GT Boundary')
plt.axvline(x=peak_index_gt, color='blue', linestyle='-', label='Peak GT')

# Plot the LWHM boundaries for prediction (left and peak indices)
plt.axvline(x=left_index_pred, color='orange', linestyle='--', label='Left Pred Boundary')
plt.axvline(x=peak_index_pred, color='orange', linestyle='-', label='Peak Pred')

# Add labels and title
plt.title("LWHM Analysis: Ground Truth vs Prediction")
plt.xlabel("Index")
plt.ylabel("Intensity")
plt.legend()

# Save the plot
plt.savefig('/date/zjz/job_1/lwhm_analysis.png')