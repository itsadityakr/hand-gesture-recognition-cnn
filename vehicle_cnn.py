import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time
from gpiozero import OutputDevice, LED, Buzzer
from luma.core.interface.serial import spi, noop
from luma.led_matrix.device import max7219
from luma.core.render import canvas
from PIL import ImageFont
import threading

# Define motor control pins (GPIO setup)
IN1 = OutputDevice(17)
IN2 = OutputDevice(27)
IN3 = OutputDevice(22)
IN4 = OutputDevice(23)

# Define LEDs for indication (GPIO pins)
WHITE_LED = LED(5)  # Forward
RED_LED = LED(6)    # Backward
YELLOW_LED_LEFT = LED(13)  # Left
YELLOW_LED_RIGHT = LED(19)  # Right
BLUE_LED = LED(12)  # Stop
GREEN_LED = LED(20)  # Valid gesture
BLUE_ERROR_LED = LED(16)  # Invalid gesture

# Buzzer setup for vehicle operation (GPIO pin)
buzzer = Buzzer(21)

# Define a function to sound the buzzer
def sound_buzzer(duration):
    buzzer.on()
    time.sleep(duration)
    buzzer.off()

# Function to start the vehicle with buzzer sound
def start_vehicle():
    for _ in range(3):  # Sound the buzzer 3 times
        sound_buzzer(0.5)  # Each sound lasts for 0.5 seconds
        time.sleep(1)  # Wait for 1 second before sounding again

# Define motor control functions
def forward():
    IN1.on()
    IN2.off()
    IN3.on()
    IN4.off()
    WHITE_LED.on()
    RED_LED.off()
    YELLOW_LED_LEFT.off()
    YELLOW_LED_RIGHT.off()
    BLUE_LED.off()
    threading.Thread(target=sound_buzzer, args=(0.5,)).start()  # Sound buzzer once when moving forward

def backward():
    IN1.off()
    IN2.on()
    IN3.off()
    IN4.on()
    RED_LED.on()
    WHITE_LED.off()
    YELLOW_LED_LEFT.off()
    YELLOW_LED_RIGHT.off()
    BLUE_LED.off()
    threading.Thread(target=sound_buzzer, args=(0.5,)).start()  # Sound buzzer once when moving backward

def left():
    IN1.off()
    IN2.off()
    IN3.on()
    IN4.off()
    YELLOW_LED_LEFT.on()
    YELLOW_LED_RIGHT.off()
    WHITE_LED.off()
    RED_LED.off()
    BLUE_LED.off()

def right():
    IN1.on()
    IN2.off()
    IN3.off()
    IN4.off()
    YELLOW_LED_RIGHT.on()
    YELLOW_LED_LEFT.off()
    WHITE_LED.off()
    RED_LED.off()
    BLUE_LED.off()

def stop():
    IN1.off()
    IN2.off()
    IN3.off()
    IN4.off()
    BLUE_LED.on()
    WHITE_LED.off()
    RED_LED.off()
    YELLOW_LED_LEFT.off()
    YELLOW_LED_RIGHT.off()

# MAX7219 initialization for LED matrix display
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=1, block_orientation=90, rotate=0)

# Load a smaller font suitable for 8x8 display
font = ImageFont.load_default()

# CNN model and MediaPipe setup
image_size = 32
model = load_model('/home/aditya/Desktop/code/cnn_model.h5')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Preprocess image function
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.flip(gray, 1)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresholded = cv2.threshold(gray, 161, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(thresholded, (image_size, image_size))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, image_size, image_size, 1))
    return reshaped, thresholded

# Map prediction to ASL letters
def predict_asl_letter(prediction):
    asl_letters = 'BFLRS'  # Backward, Forward, Left, Right, Stop
    return asl_letters[prediction]

# Function to display a letter on the MAX7219 matrix
def display_letter_on_matrix(letter):
    with canvas(device) as draw:
        # Draw the letter on the matrix using the loaded font
        draw.text((0, 0), letter, font=font, fill="white")

# Video capture setup
cap = cv2.VideoCapture(0)
previous_letter = None
asl_letter = ""  # Initialize asl_letter before the loop

# Start the vehicle with buzzer sound
start_vehicle()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the video frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Reset indicators at the beginning of each frame
    BLUE_ERROR_LED.off()  # Ensure error LEDs are off

    # If hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, c = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            # Get the bounding box for the hand
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Add margin to the bounding box
            margin = 30
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Crop and preprocess hand image
            hand_image = frame[y_min:y_max, x_min:x_max]
            preprocessed_image, resized_image = preprocess_image(hand_image)

            # Model prediction
            prediction = model.predict(preprocessed_image)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            # Map prediction to ASL letter
            asl_letter = predict_asl_letter(predicted_label)

            # Draw bounding box around hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'ASL Letter: {asl_letter} ({confidence:.2f}%)', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Show preprocessed image
            cv2.imshow('Preprocessed Image', resized_image)

            # Display the detected letter on MAX7219
            display_letter_on_matrix(asl_letter)

            # If a new letter is detected
            if asl_letter != previous_letter:
                previous_letter = asl_letter  # Update previous letter

                # Check if it's a valid gesture
                if asl_letter in ['F', 'B', 'L', 'R', 'S']:
                    GREEN_LED.on()  # Turn on green LED for valid gesture

                    if asl_letter == 'F':
                        forward()
                    elif asl_letter == 'B':
                        backward()
                    elif asl_letter == 'L':
                        left()
                    elif asl_letter == 'R':
                        right()
                    elif asl_letter == 'S':
                        stop()
                else:
                    # Invalid gesture detected
                    GREEN_LED.off()  # Turn off green LED
                    BLUE_ERROR_LED.on()  # Turn on blue LED
                    stop()  # Stop the robot if invalid gesture is detected
            else:
                # If the same letter is detected again, keep the green LED on
                if asl_letter in ['F', 'B', 'L', 'R', 'S']:
                    GREEN_LED.on()  # Keep green LED on for valid gesture
                else:
                    # No gesture detected (blue LED on)
                    BLUE_ERROR_LED.on()  # Ensure error LED is on

    else:
        # No hand detected
        GREEN_LED.off()  # Turn off green LED
        BLUE_ERROR_LED.on()  # Turn on blue LED
        stop()  # Stop the robot if no gesture is detected

    # Display frame
    cv2.imshow('ASL Recognition', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
