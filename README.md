# Hand-Gesture-using-CNN
# Work in Progress 🚧...................

## Hand Gesture Image Capture and Processing

## Overview

This project involves capturing hand gesture images using a webcam, processing these images to create black and white versions, and organizing them into separate folders. The real colored images are saved in the `images/` folder, while the processed black and white images are saved in the `dataset/` folder.

## Setup

1. **Install Dependencies**

   Ensure you have Python and the required libraries installed. Use the following commands to install the necessary packages:

   ```sh
   pip install opencv-python numpy
   ```

2. **Folder Structure**

   Ensure the following folder structure is in place:

   ```
   /your-project-directory
   ├── capture_images.py
   ├── images/
   └── dataset/
   ```

## Instructions

### 1. Capture Hand Gesture Images

1. **Run the Capture Script**

   Execute the `capture_images.py` script to start capturing images:

   ```sh
   python capture_images.py
   ```

2. **Capture Images**

   - The webcam feed will open with a blue frame displayed on the screen.
   - Place your hand inside the blue frame.
   - Press keys `1` through `6` on your keyboard to capture images. Each key press will save an image with the corresponding key number.
   - Press the `ESC` key to exit the capture process.

   - The processed black and white images will be saved in the `dataset/` folder.
   - The original colored images will remain in the `images/` folder.

## Notes

- Ensure that the `images/` and `dataset/` folders exist in your project directory before running the scripts. If they do not exist, they will be created automatically.
- The processing function applies adaptive thresholding and Gaussian blur to convert images to black and white with the hand in white and the background in black.

---