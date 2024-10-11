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

def eye_close():

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    left_eye_aspect_ratio = 0
    right_eye_aspect_ratio = 0
    threshold = 0.25
    closed_eyes_time = 0


    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) >= 2:
                # Extract the coordinates of both eyes
                (ex1, ey1, ew1, eh1) = eyes[0]
                (ex2, ey2, ew2, eh2) = eyes[1]

                # Calculate the eye aspect ratio (EAR)
                left_eye_aspect_ratio = ((ey1+eh1/2)-(ey2+eh2/2))/((ex2+ew2/2)-(ex1+ew1/2))
                right_eye_aspect_ratio = ((ey1+eh1/2)-(ey2+eh2/2))/((ex2+ew2/2)-(ex1+ew1/2))
                ear = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2

                # Check if the eye is closed
                if ear < threshold:
                    if closed_eyes_time == 0:
                        closed_eyes_time = time.time()
                    else:
                        elapsed_time = time.time() - closed_eyes_time
                        if elapsed_time >= 5:
                            print("Eyes closed for too long!")
                            playsound("./sound.mpeg")

                        # Add your notification code here
                else:
                    closed_eyes_time = 0

                    # Draw a rectangle around the eyes
                cv2.rectangle(roi_color, (ex1, ey1), (ex1+ew1, ey1+eh1), (0, 255, 0), 2)
                cv2.rectangle(roi_color, (ex2, ey2), (ex2+ew2, ey2+eh2), (0, 255, 0), 2)
                # cv2.putText(frame, "Eye is close too long", cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('r'):
            break

    cap.release()
    cv2.destroyAllWindows()


video_path = 'test.mp4'
# eye_data = cv2.VideoCapture()
object_detect(video_path)
# eye_close()

