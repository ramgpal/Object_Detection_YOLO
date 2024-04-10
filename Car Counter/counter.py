from ultralytics import YOLO
import cv2
import cvzone

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Load the video capture
cap = cv2.VideoCapture("../videos/cars.mp4")

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    results = model(frame)

    # Initialize object count
    obj_count = 0

    # Draw bounding boxes around detected objects
    for detection in results.pred:
        for obj in detection:
            obj_count += 1
            x1, y1, x2, y2 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_id = int(obj[5])
            class_name = className[class_id]
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detected objects and count
    cv2.putText(frame, f"Detected objects: {obj_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
