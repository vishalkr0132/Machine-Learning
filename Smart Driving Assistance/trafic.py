import cv2
import numpy as np

# Define the lower and upper bounds of the traffic light colors in HSV color space
red_lower = np.array([0, 120, 70])
red_upper = np.array([10, 255, 255])
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])
green_lower = np.array([50, 100, 100])
green_upper = np.array([70, 255, 255])

# Initialize the video capture
cap = cv2.VideoCapture("video.mp4")
# cap = cv2.

while True:
    # Capture a frame from the video source
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply color thresholding to isolate the traffic light
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Calculate the percentage of each color in the isolated area
    red_percent = np.sum(red_mask) / (red_mask.shape[0] * red_mask.shape[1])
    yellow_percent = np.sum(yellow_mask) / (yellow_mask.shape[0] * yellow_mask.shape[1])
    green_percent = np.sum(green_mask) / (green_mask.shape[0] * green_mask.shape[1])

    # Determine the color of the traffic light based on the highest percentage of the colors found
    if red_percent > yellow_percent and red_percent > green_percent:
        color = 'red'
    elif yellow_percent > red_percent and yellow_percent > green_percent:
        color = 'yellow'
    else:
        color = 'green'

    # Draw the rectangle around the traffic light
    cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Traffic Light Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()