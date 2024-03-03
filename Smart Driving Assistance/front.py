import cv2
import torch
import numpy as np
from scipy.spatial.distance import euclidean
from playsound import playsound
import time


def object_detect(video_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    confidence_threshold = 0.5
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        boxes = results.xyxy[0].cpu().numpy()
        scores = results.xyxy[0][:, 4].cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i][:4]
            confidence = scores[i]
            light_image = frame[int(y1):int(y2), int(x1):int(x2)]

            light_image = cv2.resize(light_image, (32, 32))
            light_image = cv2.cvtColor(light_image, cv2.COLOR_BGR2RGB)
            light_image = light_image.astype('float') / 255.0

            light_image = torch.from_numpy(np.expand_dims(light_image.transpose((2, 0, 1)), axis=0)).float()

            with torch.no_grad():
                prediction = model(light_image)

            label = np.argmax(prediction.cpu().numpy())

            confidence = np.max(prediction.cpu().numpy())

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            distances = []
            for result in results.xyxy[0]:
                xmin, ymin, xmax, ymax, confidence, class_id = result
                width = xmax - xmin
                height = ymax - ymin

                distance = euclidean((width, height), (80, 80)) *0.5
                distances.append(distance)
            cv2.putText(frame, f'{distance:.2f} m', (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if distance <20:
                playsound("./sound.mpeg")
        cv2.imshow('Traffic Light Recognition', frame)
        cv2.imshow('Traffic Light Recognition', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

video_path = 'test_video.mp4'
# eye_data = cv2.VideoCapture()
object_detect(video_path)