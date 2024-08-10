# Hand Gesture Image Capture using CNN
# 🚧🚧 IN PROGRESS 🚧🚧

---

## Overview

This project is designed to capture hand gestures using a webcam, process them with MediaPipe to detect landmarks, and save both raw gesture images and images with drawn landmarks into separate directories. The project uses OpenCV for video capture and image processing, and MediaPipe for hand landmark detection.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

Install the required Python packages using pip:

```bash
pip install opencv-python mediapipe numpy
```

## Functionality

1. **Video Capture**: Captures video from the webcam.
2. **Region of Interest (ROI)**: Extracts a portion of the frame for hand gesture recognition.
3. **Image Processing**: Applies Gaussian blur to the frame and processes the ROI to extract hand landmarks.
4. **Landmark Visualization**: Draws landmarks and edges on a blank image and displays it.
5. **Image Saving**: Saves captured gesture images and landmark images to designated directories based on key presses.

## Usage

1. **Start the Script**: Run the script with Python:

   ```bash
   python your_script_name.py
   ```

2. **Capture Gestures**: Press keys '1' to '6' to capture images of different hand gestures. Each key corresponds to a different gesture:
   - '1' - Index Finger
   - '2' - Middle Finger
   - '3' - Ring Finger
   - '4' - Pinky Finger
   - '5' - Thumb
   - '6' - Palm

   Images are saved in the `captured_256x256` directory with filenames corresponding to the number of images captured for each gesture.

3. **Exit**: Press 'Esc' to stop the video capture and terminate the script.

## Directory Structure

- `captured_256x256/`:
  - Contains subdirectories for each gesture (`index`, `middle`, `ring`, `pinky`, `thumb`, `palm`).
  - Stores captured gesture images.

- `dataset_256x256/`:
  - Contains subdirectories for each gesture (`index`, `middle`, `ring`, `pinky`, `thumb`, `palm`).
  - Stores images with drawn landmarks.

## Code Overview

- **Initialization**: Sets up MediaPipe Hands and OpenCV video capture.
- **ROI Processing**: Applies Gaussian blur to the frame and extracts the ROI.
- **Landmark Drawing**: Draws landmarks and edges on a blank image.
- **Image Saving**: Saves processed images to the specified directories.

## Troubleshooting

- **Error with OpenCV**: Ensure you have the latest version of OpenCV installed. If encountering errors with `cv2.displayOverlay`, check OpenCV documentation for support related to your specific build.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

- by Aditya Kumar