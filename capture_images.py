import cv2
import numpy as np
import os

# Define directories
captured_images_directory = 'images/'
dataset_processed_images_directory = 'dataset/'

# Ensure output folders exist
os.makedirs(captured_images_directory, exist_ok=True)
os.makedirs(dataset_processed_images_directory, exist_ok=True)

minValue = 70

def process_image(image_path, output_path):
    # Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Warning: Image not found: {image_path}")
        return

    # Apply processing function
    processed_image = func(frame)
    
    # Save the processed image
    cv2.imwrite(output_path, processed_image)
    print(f"Image saved to {output_path}")

def func(frame):    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res

def capture_and_save_images():
    cap = cv2.VideoCapture(0)
    interrupt = -1  

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Mirror the image
        
        # Define the Region of Interest (ROI)
        x1, y1, x2, y2 = 220, 10, 620, 410
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
        roi = frame[y1:y2, x1:x2]

        cv2.imshow("Frame", frame)

        # Capture and save images on key press
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27:  # ESC key to exit
            break
        for key in range(1, 7):  # Key presses from '1' to '6'
            if interrupt & 0xFF == ord(str(key)):
                # Save the image with the corresponding key number
                image_count = len([f for f in os.listdir(captured_images_directory) if f.startswith(str(key))])
                filename = os.path.join(captured_images_directory, f"{key}_{image_count}.jpg")
                cv2.imwrite(filename, roi)
                
                # Process the saved image and save to images folder
                bwf_filename = os.path.join(dataset_processed_images_directory, f"{key}_{image_count}.jpg")
                process_image(filename, bwf_filename)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_save_images()
