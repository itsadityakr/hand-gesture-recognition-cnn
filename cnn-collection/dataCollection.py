import math
import time
import cv2
import numpy as np
import os
from cvzone.HandTrackingModule import HandDetector

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Parameters
offset = 20
imgSize = 300

# Directory setup
dataset_folder = "dataset"
folders = [str(i) for i in range(1, 7)]
for folder in folders:
    os.makedirs(os.path.join(dataset_folder, folder), exist_ok=True)

# Initialize counters and folder state
counter = {str(i): 0 for i in range(1, 7)}
current_folder = None
imgCrop = None
capturing = False  # Flag to control capturing

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

    if imgCrop is not None and imgCrop.size > 0:
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # Handle folder selection and toggling capturing
    if key == ord("1"):
        current_folder = "1"
        capturing = True
    elif key == ord("2"):
        current_folder = "2"
        capturing = True
    elif key == ord("3"):
        current_folder = "3"
        capturing = True
    elif key == ord("4"):
        current_folder = "4"
        capturing = True
    elif key == ord("5"):
        current_folder = "5"
        capturing = True
    elif key == ord("6"):
        current_folder = "6"
        capturing = True
    elif key == 27:  # ESC key to stop capturing
        capturing = False
        print("Capturing stopped. Press 1-6 to resume.")
    
    # Capture and save images automatically
    if capturing and current_folder:
        folder_path = os.path.join(dataset_folder, current_folder)
        current_count = counter[current_folder]
        if imgCrop is not None and imgCrop.size > 0:
            img_path = os.path.join(folder_path, f'Image_{current_count + 1}.jpg')
            cv2.imwrite(img_path, imgWhite)
            counter[current_folder] += 1
            print(f"Saved {counter[current_folder]} images to folder {current_folder}")
        
        time.sleep(0.01)  # Add a short delay to control the capture rate

    elif key == ord("q"):  # Q key to exit
        print("Exiting Window")
        break

cap.release()
cv2.destroyAllWindows()
