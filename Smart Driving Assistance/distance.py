import torch
import cv2
import numpy as np
from scipy.spatial.distance import euclidean

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the video file
cap = cv2.VideoCapture('video.mp4')

while True:
    # Read a frame from the video file
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection on the frame
    results = model(frame)
    
    # Get the distances of the detected objects
    distances = []
    for result in results.xyxy[0]:
        xmin, ymin, xmax, ymax, confidence, class_id = result
        width = xmax - xmin
        height = ymax - ymin

        # Estimate distance based on object size
        # This is just a rough estimate and will depend on your specific use case
        distance = euclidean((width, height), (80, 80)) * 10
        distances.append(distance)
        
        # Draw a box around the detected object and show the estimated distance
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f'{distance:.2f} m', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    # Show the frame with the bounding boxes and distance labels
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()