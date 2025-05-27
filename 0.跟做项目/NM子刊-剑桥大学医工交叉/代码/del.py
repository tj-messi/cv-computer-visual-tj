import os
import glob

# 获取当前目录
current_dir = os.getcwd()

# 遍历当前目录及所有子目录中的 .pt 文件
for file_path in glob.glob(os.path.join(current_dir, "**", "*.pt"), recursive=True):
    file_name = os.path.basename(file_path)  # 获取文件名
    if file_name[-6:-3] == "000":  # 判断名字后三位是否为 "000"
        try:
            os.remove(file_path)  # 删除文件
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")