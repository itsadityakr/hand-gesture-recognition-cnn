from gpiozero import OutputDevice, Buzzer, LED, DistanceSensor, PWMOutputDevice
from time import sleep
from luma.core.interface.serial import spi, noop
from luma.core.legacy import show_message
from luma.core.legacy.font import proportional, LCD_FONT
from luma.led_matrix.device import max7219

# Motor control pins
IN1 = OutputDevice(17)
IN2 = OutputDevice(27)
IN3 = OutputDevice(22)
IN4 = OutputDevice(23)
# PWM for motor speed control
ENA = PWMOutputDevice(12)  # Motor 1 speed
ENB = PWMOutputDevice(18)  # Motor 2 speed

# LEDs
WHITE_LED = LED(5)   # Forward LED
RED_LED = LED(6)     # Backward/Stop LED
YELLOW_LED_LEFT = LED(13)  # Left Turn LED
YELLOW_LED_RIGHT = LED(19) # Right Turn LED
GREEN_LED = LED(26)  # Correct gesture LED
BLUE_LED = LED(12)   # Incorrect gesture LED

# Buzzer
BUZZER = Buzzer(16)

# Distance sensor
sensor = DistanceSensor(echo=24, trigger=25, max_distance=5)

# MAX7219 LED matrix display
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, width=8, height=8, block_orientation=0)
device.contrast(10)

# Motor control functions with speed adjustments
motor_speed = 0.5  # Default speed (50% of max speed)

def forward():
    IN1.on()
    IN2.off()
    ENA.value = motor_speed

def backward():
    IN1.off()
    IN2.on()
    ENA.value = motor_speed

def left():
    IN3.on()
    IN4.off()
    ENB.value = motor_speed

def right():
    IN3.off()
    IN4.on()
    ENB.value = motor_speed

def stop():
    IN1.off()
    IN2.off()
    IN3.off()
    IN4.off()
    ENA.value = 0
    ENB.value = 0

def increase_speed():
    global motor_speed
    if motor_speed < 1:
        motor_speed += 0.1
    print(f"Motor speed increased to {motor_speed * 100:.0f}%")

def decrease_speed():
    global motor_speed
    if motor_speed > 0:
        motor_speed -= 0.1
    print(f"Motor speed decreased to {motor_speed * 100:.0f}%")

# Update MAX7219 display
def update_display(message):
    device.clear()
    show_message(device, message, fill="white", font=proportional(LCD_FONT), scroll_delay=0.1)

# Main menu functions
def test_motors():
    global motor_speed
    motor_speed = 0.5  # Reset speed to default
    print("Motor control: Press '+' to increase speed, '-' to decrease speed, '0' to stop, 'ESC' to go back.")
    
    while True:
        k = input("Enter '+' to increase speed, '-' to decrease, or '0' to stop: ").strip()
        if k == '+':
            increase_speed()
        elif k == '-':
            decrease_speed()
        elif k == '0':
            stop()
            break
        else:
            print("Invalid input! Press '0' to stop and return.")

def test_leds():
    print("Turning on all LEDs...")
    WHITE_LED.on()
    RED_LED.on()
    YELLOW_LED_LEFT.on()
    YELLOW_LED_RIGHT.on()
    GREEN_LED.on()
    BLUE_LED.on()
    sleep(2)
    print("Turning off all LEDs...")
    WHITE_LED.off()
    RED_LED.off()
    YELLOW_LED_LEFT.off()
    YELLOW_LED_RIGHT.off()
    GREEN_LED.off()
    BLUE_LED.off()

def test_buzzer():
    print("Turning on the buzzer...")
    BUZZER.on()
    sleep(2)
    BUZZER.off()

def test_display():
    print("Turning on the MAX7219 display...")
    update_display("HELLO")
    sleep(2)
    update_display("TEST")
    sleep(2)
    update_display("DONE")

def test_sensor():
    print("Testing Distance Sensor...")
    while True:
        distance = int(sensor.distance * 100)
        print(f"Distance: {distance} cm")
        if distance < 15:
            print("Object detected! Stopping...")
            stop()
            break
        sleep(1)

# Main menu for testing components
def main_menu():
    print("Main Menu:")
    print("1. Test Motors (Use + to increase speed, - to decrease, 0 to stop)")
    print("2. Test LEDs (Turn on all LEDs)")
    print("3. Test Buzzer")
    print("4. Test MAX7219 display")
    print("5. Test Distance Sensor")
    print("ESC to exit")

    while True:
        k = input("Enter your choice: ").strip()
        
        if k == '1':
            test_motors()
        elif k == '2':
            test_leds()
        elif k == '3':
            test_buzzer()
        elif k == '4':
            test_display()
        elif k == '5':
            test_sensor()
        elif k.lower() == 'esc':
            print("Exiting...")
            break
        else:
            print("Invalid input, please try again.")

# Start the main menu
if __name__ == "__main__":
    main_menu()
