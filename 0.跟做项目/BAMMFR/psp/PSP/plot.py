import matplotlib.pyplot as plt

def plot_results(file_path):
    # 读取文本文件
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 初始化存储数据的列表
    feature_dim_used = []
    acc_values = []

    # 解析每一行数据
    for line in lines:
        if line.startswith('feature_dim_used'):
            parts = line.split()
            feature_dim_used.append(int(parts[1]))
            acc_values.append(float(parts[7]))

    return feature_dim_used, acc_values

# 文件路径列表
file_paths = ['/root/autodl-tmp/experiment/_super_resolution_1/stats.txt', 
              '/root/autodl-tmp/experiment/_super_resolution_2/stats.txt', 
              '/root/autodl-tmp/experiment/_super_resolution_3/stats.txt',
              '/root/autodl-tmp/experiment/_super_resolution_4/stats.txt']

# 对应的颜色列表和自定义标签列表
line_colors = ['blue', 'green', 'red', 'purple']
line_labels = ['Cosine similarity matches from high dimensions',
                'Cosine similarity matches from low dimensions', 
                'Euclidean distance matches from high dimensions', 
                'Euclidean distance matches from low dimensions']

# 初始化图
plt.figure(figsize=(10, 6))

# 遍历每个文件，绘制折线
for i, file_path in enumerate(file_paths):
    feature_dim_used, acc_values = plot_results(file_path)
    plt.plot(feature_dim_used, acc_values, marker='o', color=line_colors[i], label=line_labels[i])

plt.xlabel('Latent Code Layers Used')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Latent Code Layers')
plt.legend(fontsize='large')
plt.grid(True, linestyle='--', alpha=0.7)  # 添加虚线网格线

# 调整横轴刻度范围和间隔
plt.xticks(range(min(feature_dim_used), max(feature_dim_used)+1, 5))

# 调整纵轴刻度范围和间隔
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])

# 调整图例位置
plt.legend(loc='lower right')

# 调整图的边缘
plt.tight_layout()

# 保存图形到文件
plt.savefig('LineChart_optimized.png', dpi=300)

# 显示图形
plt.show()
