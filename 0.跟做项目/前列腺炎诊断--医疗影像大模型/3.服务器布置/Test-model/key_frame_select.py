import cv2
import numpy as np
import os

from PIL import Image
import numpy as np

# 设定阈值
# 阈值为均值+方差
threshold = 0

def calculate_frame_difference(frame1, frame2):
    """计算两帧之间的差异"""
    # 转换为灰度图像
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # 计算两帧之间的差异
    diff = cv2.absdiff(gray1, gray2)
    
    # 计算差异的总和
    diff_sum = np.sum(diff)
    return diff_sum

def cal_threshold (image_folder):
    now_image_path = image_folder

    #计算返回的阈值
    res_threshold = 0
    
    #前一帧标记
    prev_frame = None
    #读取所有帧
    image_files = sorted(os.listdir(image_folder))

    #差异值数组和帧数同济
    diff_list = np.zeros(1000)
    count = 0

    #遍历图像
    for idx,file_name in enumerate(image_files):
        if not file_name.endswith(".png") and not file_name.endswith(".jpg"):
            continue

        #加载当前图像
        frame = cv2.imread(os.path.join(image_folder,file_name))

        count=count+1

        if prev_frame is None:
            #第一帧跳过计科
            prev_frame = frame
            continue

        diff = calculate_frame_difference(prev_frame,frame)

        diff_list[idx] = diff

    diff_list_all = diff_list[:count]
    diff_var = np.sqrt(np.mean(diff_list_all))
    diff_avg = (np.mean(diff_list_all))

    return diff_avg+diff_var



def find_keyframes(image_folder):
    """根据阈值化方法计算关键帧"""
    # 读取图像文件夹中的所有帧
    image_files = sorted(os.listdir(image_folder))
    
    keyframes = []
    keyidxs = []
    prev_frame = None
    
    # 遍历所有的图像
    for idx, file_name in enumerate(image_files):
        if not file_name.endswith(".png") and not file_name.endswith(".jpg"):
            continue
        
        # 加载当前图像
        frame = cv2.imread(os.path.join(image_folder, file_name))
        
        if prev_frame is None:
            #第一帧作为起始帧，跳过
            prev_frame = frame
            continue
        
        # 计算当前图像与上一帧的差异
        diff_sum = calculate_frame_difference(prev_frame, frame)

        if diff_sum > threshold:
            # 如果差异超过阈值，视为关键帧
            keyframes.append(file_name)
            keyidxs.append(idx+1) #idx从0开始
        # 更新上一帧
        prev_frame = frame
    
    return keyidxs,keyframes


def get_single_keyframe(keyframes,keyidxs,image_folder_path) :
    max_white_pixel_ratio = 0

    #单个S
    single_keyframe = None
    single_keyidx = None
    for idx, image_path in enumerate(keyframes):
        # 打开图像
        image = Image.open(os.path.join(image_folder_path, image_path)).convert('L')  # 转换为灰度图

        # 将图像转换为 NumPy 数组
        image_array = np.array(image)

        # 计算白色像素的数量（255表示白色）
        white_pixels = np.sum(image_array == 255)

        # 计算总像素数量
        total_pixels = image_array.size

        # 计算白色像素占比
        white_pixel_ratio = white_pixels / total_pixels

        if white_pixel_ratio > max_white_pixel_ratio:
            max_white_pixel_ratio = white_pixel_ratio
            single_keyframe = image_path
            single_keyidx = keyidxs[idx]  # 通过索引从 keyidxs 获取对应的 keyidx
    
    return single_keyidx, single_keyframe

def main():

    image_folder_path = "./similarity-calculation/segment_png"  
    threshold = cal_threshold(image_folder_path)
    #threshold = 50000
    keyidxs,keyframes = find_keyframes(image_folder_path)

    # 输出关键帧列表
    # print("过阈值的关键帧文件：", keyframes)
    # print("过阈值的文件索引:",keyidxs)
    '''
    阈值化处理
    方差+平均值就是界限，比这个界限搞得就被阈值斩杀
    在挑选不是全黑的最大感兴趣面积
    '''

    single_keyidx,single_keyframe = get_single_keyframe(keyframes,keyidxs,image_folder_path)

    # print("感兴趣面积最大的关键帧",single_keyframe)
    # print("感兴趣面积最大的关键帧索引",single_keyidx)