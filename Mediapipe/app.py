import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize the mediapipe hands class with higher confidence thresholds.
mp_hands = mp.solutions.hands

hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                              min_detection_confidence=0.75,  # Increased detection confidence
                              min_tracking_confidence=0.75)   # Increased tracking confidence

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

def detectHandsLandmarks(image, hands, draw=True, display=True):
    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)
    
    # If landmarks are found and drawing is enabled, draw them on the image.
    if results.multi_hand_landmarks and draw:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image=image, landmark_list=hand_landmarks,
                                      connections=mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
    
    if display:
        plt.figure(figsize=[15, 15])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(image[:, :, ::-1])
        plt.title("Output")
        plt.axis('off')
    else:
        return image, results

def countFingers(image, results):
    fingers_tips_ids = {
        'THUMB': mp_hands.HandLandmark.THUMB_TIP,
        'INDEX': mp_hands.HandLandmark.INDEX_FINGER_TIP,
        'MIDDLE': mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        'RING': mp_hands.HandLandmark.RING_FINGER_TIP,
        'PINKY': mp_hands.HandLandmark.PINKY_TIP
    }
    
    hand_landmarks = results.multi_hand_landmarks[0]  # Get the first detected hand
    hand_label = results.multi_handedness[0].classification[0].label  # Get whether it's 'Left' or 'Right' hand

    # Detect finger statuses
    statuses = {finger: False for finger in fingers_tips_ids}
    thumb_tip_x = hand_landmarks.landmark[fingers_tips_ids['THUMB']].x
    thumb_ip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    thumb_status = thumb_tip_x < thumb_ip_x if hand_label == 'Right' else thumb_tip_x > thumb_ip_x
    statuses['THUMB'] = thumb_status
    
    for finger, tip_id in fingers_tips_ids.items():
        if finger == 'THUMB':
            continue
        pip_id = tip_id - 2
        finger_status = hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y
        statuses[finger] = finger_status
    
    return statuses

def palmDetected(statuses):
    if not any(statuses[finger] for finger in statuses if finger != 'THUMB'):
        print("Palm (Stop): The wheelchair stops moving.")

def indexPinkyDetected(statuses):
    if statuses['INDEX'] and statuses['PINKY'] and not statuses['MIDDLE'] and not statuses['RING']:
        print("Index Finger + Pinky (Move Forward): The wheelchair moves forward.")

def middleRingDetected(statuses):
    if statuses['MIDDLE'] and statuses['RING'] and not statuses['INDEX'] and not statuses['PINKY']:
        print("Middle Finger + Ring Finger (Move Backward): The wheelchair moves backward.")

def indexDetected(statuses):
    if statuses['INDEX'] and not any(statuses[finger] for finger in statuses if finger != 'INDEX' and finger != 'THUMB'):
        print("Index Finger (Turn Left): The wheelchair turns left.")

def pinkyDetected(statuses):
    if statuses['PINKY'] and not any(statuses[finger] for finger in statuses if finger != 'PINKY' and finger != 'THUMB'):
        print("Pinky Finger (Turn Right): The wheelchair turns right.")

def fistDetected(statuses):
    if all(not statuses[finger] for finger in statuses if finger != 'THUMB'):
        print("Fist Detected: Start")

def detectGesture(statuses, hand_landmarks):
    palmDetected(statuses)
    indexPinkyDetected(statuses)
    middleRingDetected(statuses)
    indexDetected(statuses)
    pinkyDetected(statuses)
    fistDetected(statuses)

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)

# Create named window for resizing purposes.
cv2.namedWindow('Fingers Counter', cv2.WINDOW_NORMAL)

while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue
    
    frame = cv2.flip(frame, 1)
    frame, results = detectHandsLandmarks(frame, hands_videos, display=False)
    
    if results.multi_hand_landmarks:
        statuses = countFingers(frame, results)
        detectGesture(statuses, results.multi_hand_landmarks[0])
                
    cv2.imshow('Fingers Counter', frame)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC key to exit
        break

camera_video.release()
cv2.destroyAllWindows()
