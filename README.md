# Smart Wheelchair System with Gesture Recognition

## Overview
This project presents an innovative smart wheelchair system that leverages Convolutional Neural Networks (CNN) for hand gesture recognition, allowing intuitive control through predefined gestures. Designed to enhance mobility for users, the system employs real-time gesture detection using a camera and Raspberry Pi, replacing traditional button-based controls.

### Key Functional Components

1. **Hardware Setup**
   - **GPIO Control:** Controls wheelchair movement via four motor pins:
     - IN1: Forward
     - IN2: Backward
     - IN3: Left
     - IN4: Right
   - **LEDs and Buzzer Indicators:**
     - **White LED:** Forward movement
     - **Red LED:** Backward movement or stop
     - **Yellow LEDs:** Left and right turns
     - **Green LED and Buzzer:** Correct gesture recognized
     - **Blue LED and Buzzer:** Incorrect gesture recognized
   - **Distance Sensor (HC-SR04):** Prevents collisions by detecting obstacles within 15 cm.

2. **LED Matrix Display**
   - **8x8 MAX7219 LED Matrix:** Provides visual feedback with the following indicators:
     - F: Forward
     - R: Reverse
     - <: Left turn
     - >: Right turn
     - B: Stop or wrong gesture
     - X: Obstacle detected

3. **Hand Gesture Recognition Using CNN**
   - **Gesture Set:**
     - Fist: Stops the wheelchair
     - Index + Pinky: Moves backward
     - Palm: Moves forward
     - Index + Thumb: Turns left
     - Pinky + Thumb: Turns right
   - **Data Collection:** 12000 images per gesture captured under varied conditions.
   - **CNN Model Architecture:** Convolutional, pooling, and fully connected layers for gesture classification.
   - **Model Training:** Supervised learning using labeled datasets.

4. **Hardware Integration for Movement Control**
   - **Raspberry Pi:** Captures images, runs CNN for gesture recognition, and sends commands to motors.
   - **L298 Motor Driver:** Converts Raspberry Pi signals into motor commands.
   - **Gear Motors:** Drive the wheelchair wheels for movement.

5. **Movement Control Logic**
   - **Forward:** Moves for 1 second unless an obstacle is detected.
   - **Backward:** Moves for 1 second.
   - **Left/Right Turns:** Turns for 1 second.
   - **Stop:** Activates red LED and halts movement.

6. **Gesture-to-Movement Mapping**
   | Gesture             | Action                |
   |---------------------|-----------------------|
   | Fist                | Stops the wheelchair   |
   | Index + Pinky      | Moves backward         |
   | Palm                | Moves forward          |
   | Index + Thumb      | Turns left             |
   | Pinky + Thumb      | Turns right            |

7. **Testing and Validation**
   - **Gesture Recognition Accuracy:** Evaluated under various conditions.
   - **Response Time:** Measures the delay between gesture detection and movement.
   - **System Reliability:** Long-term performance evaluation.

### Conclusion
The CNN-based smart wheelchair system demonstrates the potential of AI to improve mobility for users with disabilities. By enabling gesture-based control, the system enhances user independence and can be further developed by expanding gesture recognition, optimizing processing speeds, and adding safety features.

## Directory Structure
```plaintext
smart_wheelchair_project/
│
├── dataset/
│   ├── B/
│   │   └── (12000 images for backward gesture)
│   ├── F/
│   │   └── (12000 images for forward gesture)
│   ├── L/
│   │   └── (12000 images for left gesture)
│   ├── R/
│   │   └── (12000 images for right gesture)
│   └── S/
│       └── (12000 images for stop gesture)
│
├── dataset.py            # Captures hand gesture images
├── preprocessing.py      # Preprocesses images for training
├── train_model.py        # Defines and trains the CNN model
└── main.py              # Runs the final application
```

## File Descriptions

### 1. `dataset.py`
- **Purpose:** Captures a dataset of hand images for gestures ('B', 'F', 'L', 'R', 'S').
- **Key Steps:**
  - Initializes MediaPipe for hand detection.
  - Captures 12000 images per gesture.
  - Saves images in respective folders.

### 2. `preprocessing.py`
- **Purpose:** Preprocesses and splits the dataset for training.
- **Key Steps:**
  - Loads images, applies transformations (flipping, blurring, thresholding).
  - Splits data into training, validation, and test sets.
  - Saves preprocessed dataset as `hand_dataset.npz`.

### 3. `train_model.py`
- **Purpose:** Defines and trains a CNN to classify hand gestures.
- **Key Steps:**
  - Loads the preprocessed dataset.
  - Applies data augmentation.
  - Builds and compiles a CNN model.
  - Trains the model and saves it as `cnn_model.h5`.

### 4. `main.py`
- **Purpose:** Final run file to operate the smart wheelchair system.
- **Key Steps:**
  - Captures real-time images.
  - Uses the trained CNN model for gesture recognition.
  - Sends motor control commands based on detected gestures.

## Output
- **Preprocessed Dataset:** Saved in `valid/` folder and as `hand_dataset.npz`.
- **Trained Model:** Saved as `cnn_model.h5`.

## Installation
To get started with the project:
1. Clone the repository.
2. Install required packages using `pip install -r requirements.txt`.
3. Connect the hardware components as described in the setup section.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

