import cv2
from ultralytics import YOLO


model = YOLO("best.pt")


camera_no = 0

cap = cv2.VideoCapture(camera_no)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow(winname="yolov8_vex",mat= annotated_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()