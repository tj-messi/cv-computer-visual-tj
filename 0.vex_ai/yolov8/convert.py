import torch
from ultralytics import YOLO

# 确保你已经加载了模型
# 假设你的模型类是 MyModel，并且已经初始化过
model = YOLO(model="best.pt")  # 请替换为你的模型初始化代码

model.export(format="onnx")  # 导出为 ONNX
