import cv2
import os
import numpy as np

gesture_directory = 'HandGesture360x360/'

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

if not os.path.exists(gesture_directory):
    os.mkdir(gesture_directory)

# Create subdirectories for each gesture
gestures = ['Index', 'Middle', 'Ring', 'Pinky', 'Thumb', 'Palm']

for gesture in gestures:
    gesture_path = os.path.join(gesture_directory, gesture)
    if not os.path.exists(gesture_path):
        os.mkdir(gesture_path)

# Initialize video capture
cap = cv2.VideoCapture(0)

print("Starting video capture... Press 'Esc' to exit.")

# Set the x-coordinate for the rectangle (ROI) to be on the right side
x_start = 340  # Adjust this value based on the frame size
x_end = x_start + 300

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Exiting...")
        break

    # Flip the frame horizontally (mirror image)
    frame = cv2.flip(frame, 1)

    # Apply a strong Gaussian blur to the entire frame
    blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)

    # Extract the ROI (Region of Interest) from the frame
    roi = frame[40:300, x_start:x_end]

    # Replace the blurred region with the original (non-blurred) ROI
    blurred_frame[40:300, x_start:x_end] = roi

    # Draw the rectangle around the ROI
    cv2.rectangle(blurred_frame, (x_start, 40), (x_end, 300), (255, 255, 255), 2)
    cv2.imshow("Main Capture Window", blurred_frame)

    # Process the ROI (convert to grayscale and resize)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (360, 360))
    cv2.imshow("Right Capture Window", roi)

    count = {
        '1': len(os.listdir(os.path.join(gesture_directory, "Index"))),
        '2': len(os.listdir(os.path.join(gesture_directory, "Middle"))),
        '3': len(os.listdir(os.path.join(gesture_directory, "Ring"))),
        '4': len(os.listdir(os.path.join(gesture_directory, "Pinky"))),
        '5': len(os.listdir(os.path.join(gesture_directory, "Thumb"))),
        '6': len(os.listdir(os.path.join(gesture_directory, "Palm"))),
    }

    interrupt = cv2.waitKey(10)

    # Save the frame when a specific key is pressed and print a message
    if interrupt & 0xFF == ord('1'):
        filename = os.path.join(gesture_directory + 'Index/', str(count['1']) + '.jpg')
        cv2.imwrite(filename, roi)
        print(f"{filename}")
    if interrupt & 0xFF == ord('2'):
        filename = os.path.join(gesture_directory + 'Middle/', str(count['2']) + '.jpg')
        cv2.imwrite(filename, roi)
        print(f"{filename}")
    if interrupt & 0xFF == ord('3'):
        filename = os.path.join(gesture_directory + 'Ring/', str(count['3']) + '.jpg')
        cv2.imwrite(filename, roi)
        print(f"{filename}")
    if interrupt & 0xFF == ord('4'):
        filename = os.path.join(gesture_directory + 'Pinky/', str(count['4']) + '.jpg')
        cv2.imwrite(filename, roi)
        print(f"{filename}")
    if interrupt & 0xFF == ord('5'):
        filename = os.path.join(gesture_directory + 'Thumb/', str(count['5']) + '.jpg')
        cv2.imwrite(filename, roi)
        print(f"{filename}")
    if interrupt & 0xFF == ord('6'):
        filename = os.path.join(gesture_directory + 'Palm/', str(count['6']) + '.jpg')
        cv2.imwrite(filename, roi)
        print(f"{filename}")

    # Exit the loop when "Esc" key is pressed
    if interrupt & 0xFF == 27:  # 27 is the ASCII code for "Esc"
        print("Exiting video capture...")
        break

cap.release()
cv2.destroyAllWindows()
print("Video capture terminated.")
