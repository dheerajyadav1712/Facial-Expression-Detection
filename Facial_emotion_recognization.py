import os
import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load OpenCV's Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the detected face
        face_img = frame[y:y + h, x:x + w]

        try:
            # Perform emotion detection using DeepFace
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)

            # Extract dominant emotion
            emotion = result[0]['dominant_emotion']

            # Display detected emotion above the rectangle
            cv2.putText(frame, emotion.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except Exception as e:
            print(f"Emotion detection error: {e}")

    # Show the video feed with detected faces and emotions
    cv2.imshow("Facial Emotion Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
