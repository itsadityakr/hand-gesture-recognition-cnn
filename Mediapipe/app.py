import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Gesture recognition state
start_triggered = False

# Function to start gesture control
def start_gesture_control():
    global start_triggered
    start_triggered = True
    print("Gesture control started")

# Function for moving forward
def move_forward():
    if start_triggered:
        print("Moving forward")

# Function for moving backward
def move_backward():
    if start_triggered:
        print("Moving backward")

# Function for moving left
def move_left():
    if start_triggered:
        print("Moving left")

# Function for moving right
def move_right():
    if start_triggered:
        print("Moving right")

# Function to detect gestures based on hand landmarks
def detect_gestures(hand_landmarks):
    global start_triggered
    
    # Landmarks for fingers: [0] for wrist, [4] for thumb tip, [8] for index finger, [12] for middle finger
    # [16] for ring finger, [20] for pinky finger

    # Thumb landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    # Index finger landmarks
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    # Pinky finger landmarks
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Palm detection (when index finger MCP is higher than tip) - Start Gesture
    if index_mcp.y < index_tip.y and not start_triggered:
        start_gesture_control()
    
    if start_triggered:
        # Thumb Up - Move Forward
        if thumb_tip.y < thumb_mcp.y:
            move_forward()
        
        # Thumb Down - Move Backward
        elif thumb_tip.y > thumb_mcp.y:
            move_backward()
        
        # Index Finger - Move Left
        elif index_tip.y < index_mcp.y:
            move_left()
        
        # Pinky Finger - Move Right
        elif pinky_tip.y < pinky_tip.y:
            move_right()

# Function to create a black fingers and white background mask
def create_black_white_mask(image, results):
    # Create an empty white background
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert hand landmarks into coordinates
            for landmark in hand_landmarks.landmark:
                h, w, _ = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                # Draw the black points for fingers
                cv2.circle(mask, (cx, cy), 10, (0, 0, 0), -1)
    
    # Return the binary mask where hand is black and background is white
    return mask

# Main function to process webcam input and detect gestures
def main():
    global start_triggered
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            results = hands.process(image_rgb)
            
            # Create a black and white mask for fingers
            mask = create_black_white_mask(image, results)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # If hands are detected, draw landmarks and detect gestures
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Detect gestures based on hand landmarks
                    detect_gestures(hand_landmarks)
                    
                    # Draw hand landmarks in color
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display detection confidence on the image
                detection_confidence = results.multi_handedness[0].classification[0].score * 100  # In percentage
                cv2.putText(image, f"Accuracy: {detection_confidence:.2f}%", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Combine the original image and the black-white mask (optional display)
            combined = cv2.hconcat([image, mask_rgb])

            # Display the combined image with black/white mask and colored landmarks
            cv2.imshow('Hand Gesture Control (Black and White Mask + Color)', combined)

            # Exit if 'q' is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
