from ultralytics import YOLO
import cv2

try:
    model = YOLO('YOLO Weights/yolov8n.pt')
    result = model("images/img3.jpg", show=True)
    cv2.waitKey(0)
except Exception as e:
    print("An error occurred:", e)
