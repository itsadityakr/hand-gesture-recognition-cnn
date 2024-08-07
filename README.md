# Hand Gesture Image Capture using CNN
# 🚧🚧 IN PROGRESS 🚧🚧

This Python script uses OpenCV to capture images of hand gestures from a webcam. It saves images of different gestures in designated directories, with each gesture being associated with a specific key press.

## Prerequisites

Before running the script, make sure you have the following Python packages installed:
- `cv2` (OpenCV)
- `numpy`

You can install these packages using pip:
```bash
pip install opencv-python numpy
```

## Script Overview

1. **Setup**: Creates a directory structure to save images of different hand gestures. The directories are:
    - `HandGesture360x360/`
        - `Index/`
        - `Middle/`
        - `Ring/`
        - `Pinky/`
        - `Thumb/`
        - `Palm/`

2. **Video Capture**: Captures video from the default webcam and displays two windows:
    - **Main Capture Window**: Shows the entire frame with a rectangular region of interest (ROI) highlighted.
    - **Right Capture Window**: Displays the ROI in grayscale and resized to 360x360 pixels.

3. **ROI Extraction**: A rectangular region of interest (ROI) is extracted from the right side of the frame. This ROI is processed (converted to grayscale and resized) and displayed in the second window.

4. **Image Saving**: Depending on the key pressed ('1' to '6'), the ROI is saved as an image in the corresponding gesture directory with a filename based on the number of images already saved for that gesture.

5. **Exit**: The video capture can be terminated by pressing the "Esc" key.

## Running the Script

1. **Ensure your webcam is connected**.
2. **Run the script**:
    ```bash
    python your_script_name.py
    ```
3. **Interact with the program**:
    - Press keys `1` to `6` to save the current ROI as an image in the respective gesture directory.
    - Press `Esc` to exit the video capture.

## Example Output

When pressing key `1`, an image of the hand gesture associated with "Index" will be saved in the `HandGesture360x360/Index/` directory, named sequentially as `0.jpg`, `1.jpg`, etc., depending on the number of images already saved.

## Troubleshooting

- **No image captured**: Ensure that your webcam is correctly connected and accessible.
- **Script errors**: Verify that all required packages are installed and that there are no syntax errors in the script.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

- by Aditya Kumar