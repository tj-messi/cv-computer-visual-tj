from ultralytics import YOLO

if __name__ == '__main__':
    #构建yolov8-vex模型
    model = YOLO(model="yolov8-vex.yaml")

    #训练模型
    model.train(data="vex-data.yaml",
                epochs=10,
                imgsz=640)


