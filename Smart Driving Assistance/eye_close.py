import cv2
import time
from playsound import playsound

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

# Initialize the eye aspect ratio (EAR) variables
left_eye_aspect_ratio = 0
right_eye_aspect_ratio = 0
threshold = 0.25
closed_eyes_time = 0

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()