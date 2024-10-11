import cv2
import numpy as np

# Load the pre-trained face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Open the camera
cap = cv2.VideoCapture(0)

# Define a function to calculate the angle between the eye and the camera
def calculate_eye_angle(eye_center, face_width, face_height, camera_fov):
    # Assume a fixed distance between the camera and the person's face
    face_distance = 50  # in centimeters

    # Calculate the size of the face in pixels
    face_size_px = max(face_width, face_height)

    # Calculate the size of the face in real-world units (centimeters)
    face_size_cm = face_distance * np.tan((face_size_px / 2) * camera_fov / 180)

    # Calculate the distance from the camera to the eye in centimeters
    eye_distance_cm = (face_size_cm / face_size_px) * (face_size_px - eye_center[1])

    # Calculate the angle between the eye and the camera
    eye_angle_deg = np.arctan2(eye_center[0] - face_width/2, eye_distance_cm) * 180 / np.pi

    return eye_angle_deg

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face, detect eyes and draw a rectangle around them
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            # Calculate the eye angle
            eye_center = (x + ex + ew/2, y + ey + eh/2)
            eye_angle = calculate_eye_angle(eye_center, w, h, 60)

            # Draw the angle text
            cv2.putText(frame, f"Eye angle: {eye_angle:.2f} degrees", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()