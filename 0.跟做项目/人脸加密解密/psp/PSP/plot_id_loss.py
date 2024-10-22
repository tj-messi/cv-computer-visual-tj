import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.palplot(sns.diverging_palette(240, 10, n=4))
# plt.style.use('ggplot')



def plot_results(file_path):
    # 读取文本文件
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 初始化存储数据的列表
    feature_dim_used = []
    identity_loss = []
    
    i=0

    # 解析每一行数据
    for line in lines:
        if line.startswith('New'):
            i=i+1
            parts = line.split()
            average_score_parts = parts[4].split('+')
            average_score = float(average_score_parts[0])  # 平均分数部分
            std_deviation = float(average_score_parts[1][:-1])  # 标准差部分（去除末尾的 's'）

            feature_dim_used.append(int(i))  # 解析的数据
            identity_loss.append(1-average_score)  # 平均分数和标准差作为一个元组


    return feature_dim_used,identity_loss

# 文件路径列表
file_paths = ['/root/autodl-tmp/experiment/_super_resolution_1/inference_results/inference_metrics/stat_id.txt', 
              '/root/autodl-tmp/experiment/_super_resolution_2/inference_results/inference_metrics/stat_id.txt', 
              '/root/autodl-tmp/experiment/_super_resolution_3/inference_results/inference_metrics/stat_id.txt',
              '/root/autodl-tmp/experiment/_super_resolution_4/inference_results/inference_metrics/stat_id.txt']

line_labels = ['Cosine similarity matches from high dimensions',
                'Cosine similarity matches from low dimensions', 
                'Euclidean distance matches from high dimensions', 
                'Euclidean distance matches from low dimensions']


# 创建一个空的DataFrame用于存储数据
df = pd.DataFrame()

# Iterate over each file path and collect the data
for i, file_path in enumerate(file_paths):
    feature_dim_used, identity_loss = plot_results(file_path)
    
    # 创建临时的DataFrame，将数据添加到主DataFrame中
    df_curr = pd.DataFrame({
        'latent code layers used': feature_dim_used,
        'Identity Loss': identity_loss,
        'Metric': line_labels[i]  # 添加一个用于区分不同指标的列
    })
    
    df = pd.concat([df, df_curr])

# 绘制折线图
fig, ax = plt.subplots(figsize=(10, 6))

for label in line_labels:
    data = df[df['Metric'] == label]
    ax.plot(data['latent code layers used'], data['Identity Loss'], label=label, linewidth=2)

# 设置图表的外观
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
ax.grid(True)

# 添加标题和标签等
ax.set_title("Identity Loss vs. Latent Code Layers")
ax.set_xlabel("Latent Code Layers Used")
ax.set_ylabel("Identity Loss")
ax.legend(title='Metric', fontsize=10)

# 保存图形到文件
plt.tight_layout()
plt.savefig('optimized_line_plot_id_loss.png', dpi=300)

# 显示图形
plt.show()
    
    
    


