import cv2
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

# Main loop with keyboard control
def control_robot():
    print("Control the robot using 'W', 'A', 'S', 'D'. Press 'ESC' to exit.")

    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 640)  # Lower resolution
    camera_video.set(4, 480)
    cv2.namedWindow('Robot Camera', cv2.WINDOW_NORMAL)

    try:
        while camera_video.isOpened():
            ok, frame = camera_video.read()
            if not ok:
                continue
            
            frame = cv2.flip(frame, 1)

            cv2.imshow('Robot Camera', frame)
            
            k = cv2.waitKey(1) & 0xFF

            if k == ord('w'):  # W key for forward
                correctGestureFeedback()
                forward()
            elif k == ord('s'):  # S key for backward
                correctGestureFeedback()
                backward()
            elif k == ord('a'):  # A key for left
                correctGestureFeedback()
                left()
            elif k == ord('d'):  # D key for right
                correctGestureFeedback()
                right()
            elif k == 27:  # ESC key to exit
                stop()
                break

    except KeyboardInterrupt:
        stop()

    finally:
        camera_video.release()
        cv2.destroyAllWindows()

# Call the function to control the robot
control_robot()
