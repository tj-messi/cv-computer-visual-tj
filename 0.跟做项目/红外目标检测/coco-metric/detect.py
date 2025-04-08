import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 推理参数官方详解链接：https://docs.ultralytics.com/modes/predict/#inference-sources:~:text=of%20Results%20objects-,Inference%20Arguments,-model.predict()

# 预测框粗细和颜色修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点

if __name__ == '__main__':
    model = YOLO('/CV/xhr_project/Paper_all/CrackDetection/ultralytics-yolo11-main/runs/train_ultimate/exp6/weights/best.pt') # select your model.pt path
    model.predict(source='/CV/xhr_dataset/hrsc/images/val',  # /CV/xhr_dataset/Visdrone/VisDrone2019-DET-test-dev/images
                  imgsz=640,
                  project='runs/detect',
                  name='ours_hrsc',
                  save=False,
                  conf=0.01,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  save_txt=False, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )