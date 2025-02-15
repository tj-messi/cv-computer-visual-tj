import os
import sys

import glob


def compare_files(file1, file2):

    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        # 逐行读取并比较
        for line1, line2 in zip(f1, f2):
            if line1 != line2:
                return False
        # 检查文件长度是否一致（可能有文件较长，另一个文件较短的情况）
        return f1.readline() == f2.readline()

# files, question and answers
def main():
    train_seq = ['01', '03', '04', '05', '07', '08', '09', '10', '11', '14',
                     '15', '16', '17', '18', '19', '20', '21', '22', '23', '25']
    val_seq = ['02', '06', '12', '13', '24']
    folder_head = '/home/test/PitVQA-main/PitVQA_dataset/qa-classification/video_'
    folder_tail = '/*.txt'
    filenames = []
    for curr_seq in train_seq:
        filenames = filenames + glob.glob(folder_head + curr_seq + folder_tail)
    
    for filename in filenames:
        now_txt = filename
        prefix = (filename)[:-10]
        now_idx = int((filename.split('/')[-1]).split('.')[0])

        video=1
        step = 1
        while (now_idx - step >= 0) :
            left_idx = (now_idx - step)
            left_idx_format_num = f"{left_idx:05d}"
            left_txt_path = os.path.join(prefix,left_idx_format_num + '.txt')
            if not os.path.exists(left_txt_path) :
                break
            if compare_files(now_txt,left_txt_path) :
                video+=1
            else:
                break
            step+=1
        step = 1
        while (1) :
            right_idx = (now_idx + step)
            right_idx_format_num = f"{right_idx:05d}"
            right_txt_path = os.path.join(prefix,right_idx_format_num + '.txt')
            if not os.path.exists(right_txt_path) :
                break
            if compare_files(now_txt,right_txt_path) :
                video+=1
            else:
                break
            step+=1 
        print(video)
    

if __name__ == '__main__':
    main()