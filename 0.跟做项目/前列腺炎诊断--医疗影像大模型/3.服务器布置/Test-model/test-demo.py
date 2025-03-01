import os
import sys
import argparse
import subprocess
import re

from datetime import datetime

from PIL import Image

from sim_calculation import *
'''
2025-2-4 zjz test

存视频到Medical-SAM2-zjz/data/USVideo_final/test下
把annotation/test.txt中加入一行目录

先过segment拿到分割数据排除压缩
subprocess-test-2d.py

然后过一个similarity_cal.py

然后开始med-vita-clip

'''

def Med_sam():

    new_folders = None

    base_dir = '/root/prostate-check-zjz/Medical-SAM2-zjz/Test-Medsam2/data-test'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    else:
        print('exist!')

    dataset_base_folder = '/root/prostate-check-zjz/Medical-SAM2-zjz/data/USVideo_final/test/NoCa'
    folders_idx = len( [f for f in os.listdir(dataset_base_folder) if os.path.isdir(os.path.join(dataset_base_folder, f))])

    print(folders_idx)
    # 一次性上传一个文件，那么需要检查的文件就是第folders_count个文件！
    # 给clip做好准备
    txt_path = '/root/prostate-check-zjz/Medical-SAM2-zjz/data/USVideo_final/annotation/test.txt'
    with open(txt_path, 'w') as f:
        string_now = f'NoCa/{folders_idx},0'
        f.write(string_now)
        f.write('\n')

    now_folders = dataset_base_folder  + '/' + str(folders_idx)
    new_folders = os.path.join(base_dir,str(folders_idx))
    if not os.path.exists(new_folders):
            os.makedirs(new_folders) 
    else:
        return 0
    for file in os.listdir(now_folders):
        new_folders = os.path.join(base_dir,str(folders_idx))
        if not os.path.exists(new_folders):
            os.makedirs(new_folders) 
            
        new_folders_test = os.path.join(new_folders,'Test')
        if not os.path.exists(new_folders_test):
            os.makedirs(new_folders_test) 

        img_path = os.path.join(now_folders,file)
        img = Image.open(img_path)

        img_name,file_kind = file.split('.')
        img_save_path = os.path.join(new_folders_test,img_name)

        if not os.path.exists(img_save_path):
           os.makedirs(img_save_path)

        img_path = (img_save_path+'/'+img_name+ '_img.png')
        label_path = (img_save_path+'/'+img_name+ '_label.png')
                
        (img).save(img_path)
        (img).save(label_path)

    # 然后走一个subprocess即可
    # 脚本路径
    script_path = '/root/prostate-check-zjz/Medical-SAM2-zjz/test_2d.py'
    data_path = new_folders
    exp_name =str(folders_idx)
    # 参数列表
    args = [
        "python", script_path,
        "-net", "sam2",
        "-exp_name", f"{exp_name}",
        "-vis", "1",
        "-sam_ckpt", "/root/prostate-check-zjz/Medical-SAM2-zjz/logs/US_PROSTATE_MedSAM2_2025_01_24_03_24_07/Model/latest_epoch.pth",
        "-sam_config", "sam2_hiera_s",
        "-image_size", "1024",
        "-out_size", "512",
        "-b", "4",
        "-val_freq", "1",
        "-dataset", "US_PROSTATE",
        "-data_path", f"{data_path}"
    ]

    subprocess.run(args)

    print('idx',folders_idx)
    return folders_idx

def Med_clip(folders_idx):

    target_dir = '/root/prostate-check-zjz/Vita-CLIP-main'
    os.chdir(target_dir)

    sh_script_path = '/root/prostate-check-zjz/Vita-CLIP-main/train_scripts/data_test.sh'
    process = subprocess.Popen(['bash', sh_script_path])

    # 等待脚本完成
    process.wait()

    test_output_folder = '/root/prostate-check-zjz/Vita-CLIP-main/test_output'
    log_files = []
    latest_log = None
    for filename in os.listdir(test_output_folder):
        #print(filename)
        if filename.endswith('.log'):
            # 提取时间部分并转换为 datetime 对象
            try:
                timestamp_str = filename.split('-')[1].split('.')[0]  # 获取 '20250205_155231'
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')  # 转换为 datetime 对象
                log_files.append((filename, timestamp))
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    if log_files:
        latest_log = max(log_files, key=lambda x: x[1])  # 按时间排序，选择最晚的文件
        print(f"The latest log file is: {latest_log[0]}")
    else :
        print('not found!')
    
    # 拿到最新log
    pattern = r'output \[np\.int64\((\d+)\)\]'
    now_test_log_path = os.path.join(test_output_folder,latest_log[0])
     # 打开文件进行逐行读取
    label = None
    with open(now_test_log_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # 提取并打印标签
                label = match.group(1)
                # print(f"Found label: {label}")
                break  # 如果只需要找到第一次出现的 'output' 行，可以使用 break 停止读取

    print(label)
    # print(log_files)
    
    sim_list_dir = '/root/prostate-check-zjz/Test-model/Test-sim'
    file_path = os.path.join(sim_list_dir, f'{folders_idx}.txt')
    with open(file_path, 'a') as f:
        f.write(label)

    return label

def zjz_clip_main():
    folders_idx = Med_sam()
    if folders_idx == 0 :
        print('not new!')
    else :
        segment_base_dir = '/root/prostate-check-zjz/Test-model/logs'
        for segment_dir in os.listdir(segment_base_dir):
            parts = segment_dir.split('_')
            idx = parts[0]#序号
            if int(idx) is folders_idx :
                # 对上了序号
                segment_dir_img_path = os.path.join(segment_base_dir,segment_dir,'Samples')
                sim = get_sim(segment_dir_img_path)
                sim_list_dir = '/root/prostate-check-zjz/Test-model/Test-sim'
                file_path = os.path.join(sim_list_dir, f'{folders_idx}.txt')
                
                with open(file_path, 'w') as f:
                    for item in sim:
                        # 将 np.float64 转换为普通的浮动值并写入文件
                        f.write(f"{float(item)}\n")

    label=Med_clip(folders_idx)



    return label

if __name__ == "__main__":
    zjz_clip_main()


