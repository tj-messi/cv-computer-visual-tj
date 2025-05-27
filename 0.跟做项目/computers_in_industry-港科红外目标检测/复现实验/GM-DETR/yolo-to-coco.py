import os
import json
import numpy as np
from glob import glob
from PIL import Image

def yolo_to_coco(yolo_train_folder, yolo_val_folder, image_train_folder, image_val_folder, train_output_json, val_output_json):
    # COCO格式结构
    coco_format_train = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    coco_format_val = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_map = {}  # 类别映射
    annotation_id_train = 1
    annotation_id_val = 1
    image_id_train = 1
    image_id_val = 1

    # 获取所有类别（假设类别从0开始，不包含背景）
    with open(os.path.join(yolo_train_folder, 'classes.txt'), 'r') as f:
        categories = f.readlines()
        coco_format_train['categories'] = [{"id": idx, "name": category.strip()} for idx, category in enumerate(categories)]
        coco_format_val['categories'] = [{"id": idx, "name": category.strip()} for idx, category in enumerate(categories)]

    # 遍历训练集图像文件夹
    image_train_files = glob(os.path.join(image_train_folder, "*.png"))  # 假设图像是 .jpg 格式
    for image_file in image_train_files:
        # 生成COCO图片信息
        image_name = os.path.basename(image_file)
        coco_format_train['images'].append({
            "id": image_id_train,
            "file_name": image_name,
            "width": Image.open(image_file).width,
            "height": Image.open(image_file).height
        })

        # 获取对应的YOLO标签文件
        yolo_txt_file = os.path.join(yolo_train_folder, "labels" , "train",os.path.splitext(image_name)[0] + ".txt")
        
        if not os.path.exists(yolo_txt_file):
            continue

        # 读取YOLO标签文件
        with open(yolo_txt_file, 'r') as f:
            yolo_labels = f.readlines()

        for label in yolo_labels:
            class_id, x_center, y_center, width, height = map(float, label.strip().split())
            class_id = int(class_id)
            
            # 计算 COCO 标注框 (xmin, ymin, width, height)
            img = Image.open(image_file)
            img_width, img_height = img.size

            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            xmin = x_center - width / 2
            ymin = y_center - height / 2
            
            # 生成COCO注释信息
            coco_format_train['annotations'].append({
                "id": annotation_id_train,
                "image_id": image_id_train,
                "category_id": class_id + 1,  # 类别ID从1开始
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            annotation_id_train += 1

        image_id_train += 1

    # 遍历验证集图像文件夹
    image_val_files = glob(os.path.join(image_val_folder, "*.png"))
    for image_file in image_val_files:
        # 生成COCO图片信息
        image_name = os.path.basename(image_file)
        coco_format_val['images'].append({
            "id": image_id_val,
            "file_name": image_name,
            "width": Image.open(image_file).width,
            "height": Image.open(image_file).height
        })

        # 获取对应的YOLO标签文件
        yolo_txt_file = os.path.join(yolo_val_folder, "labels" , "val",os.path.splitext(image_name)[0] + ".txt")
        # print(yolo_txt_file)
        
        if not os.path.exists(yolo_txt_file):
            continue

        # 读取YOLO标签文件
        with open(yolo_txt_file, 'r') as f:
            yolo_labels = f.readlines()

        for label in yolo_labels:
            class_id, x_center, y_center, width, height = map(float, label.strip().split())
            class_id = int(class_id)
            
            # 计算 COCO 标注框 (xmin, ymin, width, height)
            img = Image.open(image_file)
            img_width, img_height = img.size

            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            xmin = x_center - width / 2
            ymin = y_center - height / 2
            
            # 生成COCO注释信息
            coco_format_val['annotations'].append({
                "id": annotation_id_val,
                "image_id": image_id_val,
                "category_id": class_id + 1,  # 类别ID从1开始
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            annotation_id_val += 1

        image_id_val += 1

    # 保存为两个JSON文件
    with open(train_output_json, 'w') as f_train:
        json.dump(coco_format_train, f_train, indent=4)

    with open(val_output_json, 'w') as f_val:
        json.dump(coco_format_val, f_val, indent=4)


# 使用示例：
yolo_train_folder = "/CV/zzx/zjz-dataset/M3FD-yolo"  # YOLO训练标签文件夹
yolo_val_folder = "/CV/zzx/zjz-dataset/M3FD-yolo"  # YOLO验证标签文件夹
image_train_folder = "/CV/zzx/zjz-dataset/M3FD-yolo/images/train"  # 训练集图像文件夹
image_val_folder = "/CV/zzx/zjz-dataset/M3FD-yolo/images/val"  # 验证集图像文件夹
train_output_json = "/CV/zzx/zjz-dataset/M3FD-yolo/train_json"  # 输出的训练集COCO格式JSON文件路径
val_output_json = "/CV/zzx/zjz-dataset/M3FD-yolo/val_json"  # 输出的验证集COCO格式JSON文件路径

yolo_to_coco(yolo_train_folder, yolo_val_folder, image_train_folder, image_val_folder, train_output_json, val_output_json)
