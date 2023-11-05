import cv2 as cv
from gtts import gTTS
import pygame

# Loading the cascade classifier for stop signs
dataset = cv.CascadeClassifier('stop_data.xml')

# Connecting to the webcam where '0' represents default camera
cap = cv.VideoCapture(0)

while True:
    # Reading a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Converting the frame to grayscale for detection
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Performing stop sign detection
    found = dataset.detectMultiScale(frame_gray, minSize=(20, 20))

    for (x, y, width, height) in found:
        cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
        font_color = (255, 255, 255)  # White font color (BGR format)
        cv.putText(frame, "Stop Sign", (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, font_color, 2)

        # Text to speech message
        message = "Stop Sign Detected"
        speech = gTTS(message)
        pygame.mixer.init()
        pygame.mixer.music.load("stop_sign.mp3")
        pygame.mixer.music.play()
        pygame.time.delay(2000) 

    # Display the frame
    cv.imshow('Stop Sign Detection', frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
