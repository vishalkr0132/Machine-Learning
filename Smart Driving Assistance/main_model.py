import cv2
import numpy as np

# Load the YOLO model
net = cv2.dnn.readNet("yolo.weights", "yolo.cfg")

# Get the class labels
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open the video file
cap = cv2.VideoCapture("video.mp4")

# Loop over the frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Stop if end of video file
    if not ret:
        break

    # Preprocess the input
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # Run the YOLO model
    net.setInput(blob)
    outs = net.forward()

    # Loop over the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Draw bounding box and label
                height, width, channels = frame.shape
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, classes[class_id], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Video", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()