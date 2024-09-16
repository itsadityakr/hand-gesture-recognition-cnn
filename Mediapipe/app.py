import cv2
import mediapipe as mp

# Initialize mediapipe hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to classify the hand gesture
def classify_hand_gesture(landmarks, handedness):
    # Names for fingers based on MediaPipe indices
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    
    # Tips of each finger (MediaPipe Landmark IDs)
    finger_tips = [4, 8, 12, 16, 20]
    
    # Folded status for fingers (1 if folded, 0 if extended)
    folded_status = []
    
    # Check each finger's status (extended or folded)
    for i, tip in enumerate(finger_tips):
        # Compare the finger tip position to the corresponding MCP (base knuckle)
        if landmarks[tip].y < landmarks[tip - 2].y:  # If the tip is above the base knuckle, the finger is extended
            folded_status.append(0)  # 0 means the finger is extended
        else:
            folded_status.append(1)  # 1 means the finger is folded
    
    # Identify the gesture
    if all(folded_status):  # All fingers are folded
        return "Fist"
    elif all([status == 0 for status in folded_status]):  # All fingers are extended
        return "Palm"
    elif folded_status == [0, 1, 1, 1, 1]:  # Only thumb extended
        if landmarks[4].x < landmarks[3].x:  # Check if thumb is pointing up
            return "Thumb Up"
        else:
            return "Thumb Down"
    elif folded_status == [0, 0, 1, 1, 1]:  # Thumb and index finger extended
        return "Thumb + Index"
    else:
        active_fingers = [finger_names[i] for i, status in enumerate(folded_status) if status == 0]
        return "Active Fingers: " + ", ".join(active_fingers)

# Initialize webcam and hand detection
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue
        
        # Convert the image to RGB for processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        
        # Convert back to BGR for OpenCV operations
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract handedness (Left or Right)
                label = handedness.classification[0].label
                # Get the list of landmarks
                landmarks = hand_landmarks.landmark
                # Classify the gesture
                gesture = classify_hand_gesture(landmarks, label)
                # Display gesture on the image
                cv2.putText(image, f'{label} Hand: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display the resulting image
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()
