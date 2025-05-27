import os
import pandas as pd
import numpy as np

def process_metrics_files(root_directory):
    # 存储所有Average行的数据
    rmse_list = []
    ssim_list = []
    cc_list = []

    # 遍历根目录下的所有子文件夹
    for i in range(0, 981, 20):
        subdir = os.path.join(root_directory, f"G{i}-{i+20}")
        files = os.listdir(subdir)
        for filename in files:
            if filename == 'metrics.csv':
                filepath = os.path.join(subdir, filename)
                
                try:
                    # 读取CSV文件
                    df = pd.read_csv(filepath)
                    
                    # 找到Average行
                    avg_row = df[df['SampleID'] == 'Average']
                    
                    # 提取RMSE, SSIM, CC值
                    rmse_list.append(float(avg_row['RMSE'].values[0]))
                    ssim_list.append(float(avg_row['SSIM'].values[0]))
                    cc_list.append(float(avg_row['CC'].values[0]))
                
                except Exception as e:
                    print(f"处理 {filepath} 时出错: {e}")

    # 计算平均值
    avg_rmse = np.mean(rmse_list)
    avg_ssim = np.mean(ssim_list)
    avg_cc = np.mean(cc_list)

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'Metric': ['RMSE', 'SSIM', 'CC'],
        'Average': [avg_rmse, avg_ssim, avg_cc]
    })

    # 保存结果到CSV
    output_dir = os.path.join(root_directory, 'Fianl_metrics/metrics_average.csv')
    result_df.to_csv(output_dir, index=False)
    
    print(f"处理完成，共找到 {len(rmse_list)} 个metrics.csv文件")
    print("结果已保存到 metrics_average.csv")
    print(f"RMSE平均值: {avg_rmse}")
    print(f"SSIM平均值: {avg_ssim}")
    print(f"CC平均值: {avg_cc}")

# 使用示例
root_directory = r'/home/hanyu/hanyu_code/NC-code3.18/TEST_Result/Ours-Xenium5k_ovaryovariancancer/10X'  # 替换为实际的根目录路径
process_metrics_files(root_directory)
