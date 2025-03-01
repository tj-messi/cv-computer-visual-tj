import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

from key_frame_select import *



'''
aSimdix + bSimIou
a<b 
'''
a=0.7
b=0.3 

def cal_idxsimilarity(single_keyidx,nowidx,len):
    return a*((len-abs(nowidx-single_keyidx))/len)

 
def cal_Iousimilarity(single_keyframe, nowframe):

    # 二值化处理，白色区域为255，其他区域为0
    binary1 = (single_keyframe == 255).astype(np.uint8)
    binary2 = (nowframe == 255).astype(np.uint8)

    # 计算交集 (白色区域的交集)
    intersection = np.sum(binary1 & binary2)

    # 计算并集 (白色区域的并集)
    union = np.sum(binary1 | binary2)

    # 计算IoU (交集 / 并集)
    iou = intersection / union if union != 0 else 0

    return iou


def cal_similarity(image_folder_path,single_keyframe,single_keyidx):
    '''
    输入图片路径image_folder_path
    输入单帧最关键帧single_keyframe
    '''    

    # 读取图像文件夹中的所有帧
    image_files = sorted(os.listdir(image_folder_path))
    
    length=len(image_files)

    similarity = []

    # 遍历图像
    for idx, file_name in enumerate(image_files):
        if not file_name.endswith(".png") and not file_name.endswith(".jpg"):
            continue

        # 加载当前图像
        frame = cv2.imread(os.path.join(image_folder_path, file_name))

        if single_keyframe is None :
            similarity.append(np.float64(0))
            continue

        keyframe = cv2.imread(os.path.join(image_folder_path, single_keyframe))

        similarity_now = cal_idxsimilarity(single_keyidx,idx,length)+b*cal_Iousimilarity(keyframe,frame)

        similarity.append(similarity_now)

        # 将图像转换为 NumPy 数组
        image_array = np.array(frame)
        white_pixels = np.sum(image_array == 255)
        if white_pixels==0:
            similarity[idx]=0
    
    return similarity


def get_sim(image_folder_path):

    # print(image_folder_path)
    #读取所有帧
    image_files = sorted(os.listdir(image_folder_path))
    threshold = cal_threshold(image_folder_path)

    keyidxs,keyframes = find_keyframes(image_folder_path)
    # 输出关键帧列表
    # print("过阈值的关键帧文件：", keyframes)
    # print("过阈值的文件索引:",keyidxs)

    single_keyidx,single_keyframe = get_single_keyframe(keyframes,keyidxs,image_folder_path)
    # 输出唯一关键帧
    # print("感兴趣面积最大的关键帧",single_keyframe)
    # print("感兴趣面积最大的关键帧索引",single_keyidx)

    
    sim=cal_similarity(image_folder_path,single_keyframe,single_keyidx)
    #print(sim)
    return sim

'''#可视化
plt.figure(figsize=(10, 6))
plt.bar(range(len(sim)), sim)
plt.xlabel('Index')
plt.ylabel('Similarity Score')
plt.title('Similarity Scores Distribution')
# 显示表格
df = pd.DataFrame(sim, columns=["Similarity Scores"])
plt.table(cellText=df.values, colLabels=df.columns, loc='bottom', cellLoc='center', colLoc='center')
plt.show()'''