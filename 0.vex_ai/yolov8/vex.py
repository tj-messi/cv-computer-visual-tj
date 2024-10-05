from ultralytics import YOLO

if __name__ == '__main__':
    #构建yolov8-vex模型
    model = YOLO(model="yolov8.yaml")

    #训练模型
    model.train(data="vex-data.yaml",
                epochs=100,
                imgsz=640)
    #epoch=100 训练轮次要够大才能进行收敛
    #imgsz=640 训练图片的尺寸要够大，否则训练效果会很差


