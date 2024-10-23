import cv2
import mediapipe as mp
from gpiozero import OutputDevice, Buzzer, LED, DistanceSensor
from time import sleep
from luma.core.interface.serial import spi, noop
from luma.core.legacy import show_message
from luma.core.legacy.font import proportional, LCD_FONT
from luma.led_matrix.device import max7219

# Pin Definitions
IN1 = OutputDevice(17)
IN2 = OutputDevice(27)
IN3 = OutputDevice(22)
IN4 = OutputDevice(23)

WHITE_LED = LED(5)   # Forward LED
RED_LED = LED(6)     # Backward/Stop LED
YELLOW_LED_LEFT = LED(13)  # Left Turn LED
YELLOW_LED_RIGHT = LED(19) # Right Turn LED
GREEN_LED = LED(26)  # Correct gesture LED
BLUE_LED = LED(12)   # Incorrect gesture LED

sensor = DistanceSensor(echo=24, trigger=25, max_distance=5)
BUZZER = Buzzer(16)

# LED matrix display
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, width=8, height=8, block_orientation=0)
device.contrast(10)

# Track movement state
moving_forward = False

def update_display(message):
    device.clear()
    show_message(device, message, fill="white", font=proportional(LCD_FONT), scroll_delay=0.1)

def check_distance():
    distance = int(sensor.distance * 100)
    if distance < 15:
        stop()
        update_display("X")
        return True
    return False

def forward():
    global moving_forward
    if check_distance():
        return
    moving_forward = True
    IN1.on()
    IN2.off()
    IN3.on()
    IN4.off()
    WHITE_LED.on()
    update_display("F")
    sleep(1)  # Move forward for 1 second
    WHITE_LED.off()
    stop()
    

def backward():
    IN1.off()
    IN2.on()
    IN3.off()
    IN4.on()
    RED_LED.on()
    update_display("R")
    sleep(1)  # Reduce time for backward movement
    RED_LED.off()
    stop()

def left():
    IN1.off()
    IN2.off()
    IN3.on()
    IN4.off()
    YELLOW_LED_LEFT.on()
    update_display("<")
    sleep(1)  # Reduce time for turning
    YELLOW_LED_LEFT.off()
    stop()

def right():
    IN1.on()
    IN2.off()
    IN3.off()
    IN4.off()
    YELLOW_LED_RIGHT.on()
    update_display(">")
    sleep(1)
    YELLOW_LED_RIGHT.off()
    stop()

def stop():
    global moving_forward
    moving_forward = False
    IN1.off()
    IN2.off()
    IN3.off()
    IN4.off()
    RED_LED.on()
    WHITE_LED.off()
    BUZZER.off()
    BLUE_LED.on()
    update_display("B")

# Correct gesture feedback (green LED + buzzer)
def correctGestureFeedback():
    GREEN_LED.on()
    BUZZER.on()
    sleep(0.5)  # 0.5 second feedback
    BUZZER.off()
    GREEN_LED.off()

# Incorrect gesture feedback (blue LED + buzzer)
def wrongGestureFeedback():
    BLUE_LED.on()
    BUZZER.on()
    update_display("WG")  # Display 'B' for a wrong gesture
    sleep(2)  # 2 second feedback
    BUZZER.off()
    BLUE_LED.off()

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                              min_detection_confidence=0.6,  # Lower detection threshold
                              min_tracking_confidence=0.6)   # Lower tracking confidence
mp_drawing = mp.solutions.drawing_utils

def detectHandsLandmarks(image, hands):
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    return results

def countFingers(results):
    fingers_tips_ids = {
        'THUMB': mp_hands.HandLandmark.THUMB_TIP,
        'INDEX': mp_hands.HandLandmark.INDEX_FINGER_TIP,
        'MIDDLE': mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        'RING': mp_hands.HandLandmark.RING_FINGER_TIP,
        'PINKY': mp_hands.HandLandmark.PINKY_TIP
    }
    hand_landmarks = results.multi_hand_landmarks[0]
    hand_label = results.multi_handedness[0].classification[0].label
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

# Gesture detection functions
def indexPinkyDetected(statuses):
    return statuses['INDEX'] and statuses['PINKY'] and not statuses['MIDDLE'] and not statuses['RING']

def indexThumbDetected(statuses):
    return statuses['INDEX'] and statuses['THUMB'] and not statuses['MIDDLE'] and not statuses['RING'] and not statuses['PINKY']

def pinkyThumbDetected(statuses):
    return statuses['PINKY'] and statuses['THUMB'] and not statuses['INDEX'] and not statuses['MIDDLE'] and not statuses['RING']

def middleRingDetected(statuses):
    return statuses['MIDDLE'] and statuses['RING'] and not statuses['INDEX'] and not statuses['PINKY']

def palmDetected(statuses):
    return not any(statuses[finger] for finger in statuses if finger != 'THUMB')

# Track the last detected gesture to prevent multiple executions
last_detected_gesture = None

def gesture_action(gesture_name, action_function):
    global last_detected_gesture
    if last_detected_gesture == gesture_name:
        return  # Ignore repeated gesture
    correctGestureFeedback()  # Give feedback for a correct gesture
    action_function()  # Execute the action
    last_detected_gesture = gesture_name  # Update the last detected gesture

# Main loop with optimized settings
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 640)  # Lower resolution
camera_video.set(4, 480)
cv2.namedWindow('Fingers Counter', cv2.WINDOW_NORMAL)

frame_skip = 1  # Only process every 4th frame to reduce load
frame_count = 0

try:
    while camera_video.isOpened():
        ok, frame = camera_video.read()
        if not ok:
            continue
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        if frame_count % frame_skip == 0:
            results = detectHandsLandmarks(frame, hands_videos)
            
            if results and results.multi_hand_landmarks:
                statuses = countFingers(results)
                
                if indexPinkyDetected(statuses):
                    gesture_action("Index + Pinky", forward)
                elif indexThumbDetected(statuses):
                    gesture_action("Index + Thumb", left)
                elif pinkyThumbDetected(statuses):
                    gesture_action("Pinky + Thumb", right)
                elif middleRingDetected(statuses):
                    gesture_action("Middle + Ring", backward)
                elif palmDetected(statuses):
                    print("Palm detected: Stopping.")
                    stop()  # Call stop() when the palm gesture is detected
                else:
                    # Gesture not recognized, provide feedback
                    wrongGestureFeedback()  # Feedback for incorrect gesture
                    last_detected_gesture = None  # Reset the last gesture when wrong gesture occurs
            
            else:
                # No hand detected
                BLUE_LED.on()  # Turn on blue LED when no gesture detected
                last_detected_gesture = None  # Reset when no hand landmarks are detected

        else:
            BLUE_LED.off()  # Turn off blue LED if hand detected

        cv2.imshow('Fingers Counter', frame)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key to exit
            break

except KeyboardInterrupt:
    stop()

finally:
    camera_video.release()
    cv2.destroyAllWindows()
