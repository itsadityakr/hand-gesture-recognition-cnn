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

# LED pins
WHITE_LED = LED(5)   # Forward LED
RED_LED = LED(6)     # Backward/Stop LED
YELLOW_LED_LEFT = LED(13)  # Left Turn LED
YELLOW_LED_RIGHT = LED(19) # Right Turn LED
GREEN_LED = LED(26)  # Correct Key Entry LED
BLUE_LED = LED(21)   # Incorrect Key Entry LED

# Ultrasonic sensor
sensor = DistanceSensor(echo=24, trigger=25, max_distance=5)

# Piezo Buzzer pin
BUZZER = Buzzer(16)

# Setup the LED matrix display
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, width=8, height=8, block_orientation=0)
device.contrast(10)

# Track movement state
moving_forward = False

def update_display(message):
    device.clear()
    show_message(device, message, fill="white", font=proportional(LCD_FONT), scroll_delay=0.1)

def check_distance():
    distance = int(sensor.distance * 100)  # Measure the distance in centimeters
    if distance < 15:
        stop()  # Stop moving forward if an object is detected within 15 cm
        update_display("X")  # Display 'X' when an object is detected
        return True  # Object detected
    elif distance < 50:
        stop()  # Stop moving forward if an object is detected within 100 cm
        update_display("X")  # Display 'X' when an object is detected
        return True  # Object detected
    return False  # No object detected within the range

def forward():
    global moving_forward
    # Initial distance check before moving forward
    if check_distance():
        return  # If an object is detected, do not move

    moving_forward = True
    IN1.on()
    IN2.off()
    IN3.on()
    IN4.off()
    WHITE_LED.on()  # Turn on forward LED
    RED_LED.off()   # Turn off red LED
    update_display("^")  # Display 'F' while moving forward

    while moving_forward:
        if check_distance():  # Check distance frequently
            break  # Stop if an object is detected
        sleep(0.05)  # Check distance every 0.2 seconds for quick response
    stop()

def backward():
    IN1.off()
    IN2.on()
    IN3.off()
    IN4.on()
    WHITE_LED.off()  # Turn off forward LED
    RED_LED.on()     # Turn on backward LED
    update_display("B")  # Display 'B' when moving backward
    sleep(2)         # Move backward for 2 seconds
    stop()

def left():
    IN1.off()
    IN2.off()
    IN3.on()
    IN4.off()
    YELLOW_LED_LEFT.on()   # Turn on left turn LED
    YELLOW_LED_RIGHT.off() # Turn off right turn LED
    update_display("<")  # Display 'L' when turning left
    sleep(2)               # Turn left for 2 seconds
    stop()

def right():
    IN1.on()
    IN2.off()
    IN3.off()
    IN4.off()
    YELLOW_LED_RIGHT.on()   # Turn on right turn LED
    YELLOW_LED_LEFT.off() # Turn off left turn LED
    update_display(">")  # Display 'R' when turning right
    sleep(2)               # Turn right for 2 seconds
    stop()

def stop():
    global moving_forward
    moving_forward = False
    IN1.off()
    IN2.off()
    IN3.off()
    IN4.off()
    RED_LED.on()     # Turn on red LED when stopped
    WHITE_LED.off()  # Turn off white LED
    BUZZER.off()     # Turn off buzzer
    update_display("-")  # Display '-' when stopped

# Main Loop
try:
    while True:
        command = input("Enter command (w: forward, s: backward, a: left, d: right, space: stop, q: quit): ").lower()
        
        # List of valid commands
        valid_commands = ["w", "s", "a", "d", " ", "q"]

        if command in valid_commands:
            if command == "w":
                forward()
            elif command == "s":
                backward()
            elif command == "a":
                left()
            elif command == "d":
                right()
            elif command == " ":
                stop()
                print("Stopping...")
            elif command == "q":
                stop()
                break
        else:
            print("Invalid command!")

except KeyboardInterrupt:
    stop()
