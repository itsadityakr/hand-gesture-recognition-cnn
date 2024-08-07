import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,  # Higher confidence for detection
    min_tracking_confidence=0.5    # Higher confidence for tracking
)
mp_drawing = mp.solutions.drawing_utils

# Define the directory structure
captured_directory = 'captured_256x256'
dataset_directory = 'dataset_256x256'
gestures = ['index', 'middle', 'ring', 'pinky', 'thumb', 'palm']

# Create subdirectories for gestures in dataset_256x256
for gesture in gestures:
    gesture_path = os.path.join(dataset_directory, gesture)
    if not os.path.exists(gesture_path):
        os.makedirs(gesture_path)

# Create the directory for captured images
if not os.path.exists(captured_directory):
    os.makedirs(captured_directory)

# Initialize video capture
cap = cv2.VideoCapture(0)
print("Starting video capture... Press 'Esc' to exit.")

# Set the x-coordinate for the rectangle (ROI) to be on the right side
x_start = 340  # Adjust this value based on the frame size
x_end = x_start + 300

# Initialize a blank image for the landmark window
edges_image = np.zeros((300, 300, 3), dtype=np.uint8)  # Black background

# Function to draw landmarks and edges on a blank image
def draw_landmarks_on_blank_image(image_shape, landmarks):
    blank_image = np.zeros(image_shape, dtype=np.uint8)  # Black background
    edge_colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255)     # Yellow
    ]
    hand_connections = mp_hands.HAND_CONNECTIONS

    for i, (start_idx, end_idx) in enumerate(hand_connections):
        color = edge_colors[i % len(edge_colors)]
        start = (int(landmarks[start_idx].x * image_shape[1]), 
                 int(landmarks[start_idx].y * image_shape[0]))
        end = (int(landmarks[end_idx].x * image_shape[1]), 
               int(landmarks[end_idx].y * image_shape[0]))
        cv2.line(blank_image, start, end, color, 2)

    for landmark in landmarks:
        x = int(landmark.x * image_shape[1])
        y = int(landmark.y * image_shape[0])
        cv2.circle(blank_image, (x, y), 5, (255, 255, 255), -1)
    
    return blank_image

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Exiting...")
        break

    frame = cv2.flip(frame, 1)

    # Apply a strong Gaussian blur to the entire frame
    blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)

    # Extract the ROI (Region of Interest) from the frame
    roi = frame[40:300, x_start:x_end]

    # Replace the blurred region with the original (non-blurred) ROI
    blurred_frame[40:300, x_start:x_end] = roi

    # Draw the rectangle around the ROI
    cv2.rectangle(blurred_frame, (x_start, 40), (x_end, 300), (255, 255, 255), 2)

    # Process the ROI for hand landmarks
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = hands.process(roi_rgb)

    # Convert ROI to grayscale and resize for display
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, (256, 256))

    # Display the "Right Capture Window"
    cv2.imshow("Right Capture Window", roi_resized)

    # Update the landmark window
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            edges_image = draw_landmarks_on_blank_image((300, 300, 3), hand_landmarks.landmark)
    else:
        edges_image = np.zeros((300, 300, 3), dtype=np.uint8)  # Blank image if no hand is detected

    # Resize the landmark window image
    edges_resized = cv2.resize(edges_image, (256, 256))

    # Display the "Landmark Window"
    cv2.imshow("Landmark Window", edges_resized)

    # Display the "Main Capture Window"
    cv2.imshow("Main Capture Window", blurred_frame)

    interrupt = cv2.waitKey(10) & 0xFF

    if interrupt in range(ord('1'), ord('7')):  # Check if key is between '1' and '6'
        gesture = gestures[interrupt - ord('1')]

        captured_path = os.path.join(captured_directory, gesture)
        if not os.path.exists(captured_path):
            os.makedirs(captured_path)
        filename = os.path.join(captured_path, f'{len(os.listdir(captured_path))}.jpg')
        cv2.imwrite(filename, roi_resized)
        print(f"{filename}")

        edge_path = os.path.join(dataset_directory, gesture)
        if not os.path.exists(edge_path):
            os.makedirs(edge_path)
        edge_filename = os.path.join(edge_path, f'{len(os.listdir(edge_path))}.jpg')
        cv2.imwrite(edge_filename, edges_resized)
        print(f"{edge_filename}")

    if interrupt == 27:  # Press 'Esc' to exit
        print("Exiting video capture...")
        break

cap.release()
cv2.destroyAllWindows()
print("Video capture terminated.")
