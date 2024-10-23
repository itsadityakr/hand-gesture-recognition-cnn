import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time

# Define the size to which the hand image will be resized for the CNN model
image_size = 32

# To keep track of the previously predicted recognition
previous_recognition = None

# Load the trained gesture recognition model
model = load_model('cnn_model.h5')

# Initialize MediaPipe Hand Detector with configurations for real-time detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Define the function to preprocess the hand image before sending it to the CNN model
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.flip(gray, 1)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresholded = cv2.threshold(gray, 161, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(thresholded, (image_size, image_size))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, image_size, image_size, 1))
    return reshaped, thresholded

# Define a function to map the CNN model's prediction to the corresponding gesture recognition
def predict_gesture_recognition(prediction):
    gesture_recognition = 'BFLRS'  # Map for backward, forward, left, right, stop
    return gesture_recognition[prediction]

# Define the functions for movement control
def backward():
    print("Backward")

def forward():
    print("Forward")

def left():
    print("Left")

def right():
    print("Right")

def stop():
    print("Stop")

# Function to display the detected gesture on the video frame
def display_gesture_detected(frame, gesture):
    cv2.putText(frame, f'Gesture Detected: {gesture}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Function to display "No gesture detected" when no gesture is detected
def display_no_gesture_detected(frame):
    cv2.putText(frame, 'No Gesture Detected', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Start video capture for live gesture recognition
cap = cv2.VideoCapture(0)

# Initialize variables for gesture recognition and time tracking
gesture_recognition = ""
last_recognition_time = 0

# Main loop for video capture and gesture recognition
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the video frame to RGB as MediaPipe requires this color format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands using MediaPipe
    result = hands.process(frame_rgb)

    # Check if hands are detected in the frame
    if result.multi_hand_landmarks:
        # Loop through detected hands
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape

            # Initialize variables for the bounding box around the hand
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            # Loop through hand landmarks to find the min/max coordinates for the bounding box
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Add a margin to the bounding box for better cropping
            margin = 30
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Crop the region of the hand from the frame
            hand_image = frame[y_min:y_max, x_min:x_max]

            # Preprocess the cropped hand image
            preprocessed_image, resized_image = preprocess_image(hand_image)

            # Use the trained model to predict the gesture recognition
            prediction = model.predict(preprocessed_image)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction) * 100  # Calculate the confidence of the prediction

            # Map the predicted label to the corresponding gesture recognition
            gesture_recognition = predict_gesture_recognition(predicted_label)

            # Draw a bounding box around the detected hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Show the preprocessed hand image in a separate window
            cv2.imshow('Preprocessed Image', resized_image)

            # Display the detected gesture on the frame
            display_gesture_detected(frame, gesture_recognition)

            # Execute the appropriate function based on the gesture recognition
            if gesture_recognition == 'B':
                backward()
            elif gesture_recognition == 'F':
                forward()
            elif gesture_recognition == 'L':
                left()
            elif gesture_recognition == 'R':
                right()
            elif gesture_recognition == 'S':
                stop()
    else:
        # If no hand gesture is detected, display "No Gesture Detected"
        display_no_gesture_detected(frame)

    # Display the video frame with hand detection and gesture recognition
    cv2.imshow('Gesture Recognition', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, exit the loop
    if key == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
