# Hand Gesture using CNN

## Project Structure

```plaintext
smart_wheelchair/
├── dataset/
│   ├── B/                # Images for backward gesture
│   ├── F/                # Images for forward gesture
│   ├── L/                # Images for left turn gesture
│   ├── R/                # Images for right turn gesture
│   ├── S/                # Images for stop gesture
├── dataset.py            # Captures dataset using webcam and MediaPipe
├── preprocessing.py      # Preprocesses and splits the dataset
├── train_model.py        # Trains the CNN model
├── sign_detector.py      # Real-time gesture recognition and wheelchair control
├── cnn_model.h5          # Trained CNN model
├── hand_dataset.npz      # Preprocessed dataset
└── README.md             # Project documentation
```

## Key Features

- **Gesture-Based Navigation**: 
  - **Move Forward**: ✋ Palm 
  - **Move Backward**: 🤘 Index + Pinky 
  - **Turn Left**: 👆 Index + Thumb 
  - **Turn Right**: 🤙 Pinky + Thumb 
  - **Stop**: ✊ Fist
- **Real-Time Gesture Recognition**: 
  - A CNN model processes live webcam images to detect hand gestures and translates them into movement commands in real time.


## 1. `dataset.py` - Dataset Collection

This script is responsible for creating a dataset of hand gesture images using a webcam and **MediaPipe** for real-time hand tracking. 

### Key Steps:

- **Initialization**:
  - **MediaPipe** is initialized to detect hand landmarks in real time.

- **Capture Images**:
  - Opens a live webcam feed.
  - Detects hands and captures the region of interest (ROI).

- **Image Storage**:
  - Captures images based on the gesture the user wants to record.
  - Saves them in folders corresponding to specific gestures (e.g., `B`, `F`, `L`, `R`, `S`).

### Example of Usage:
```bash
python dataset.py
```
This command starts capturing images of hand gestures, which are stored in a structured dataset.

---

## 2. `preprocessing.py` - Data Preprocessing and Splitting

This script prepares the dataset for training by applying transformations and splitting it into training, validation, and test sets.

### Key Steps:

- **Image Preprocessing**:
  - Uses **OpenCV** to load images:
    - Resize images to **32x32 pixels**.
    - Convert images to **grayscale**.
    - Apply **thresholding** and **normalization**.

- **Data Augmentation**:
  - Enhances the dataset by applying transformations like flipping, rotation, and zooming.

- **Dataset Splitting**:
  - Splits the dataset into:
    - **70% Training Set**
    - **20% Validation Set**
    - **10% Test Set**

- **Saving Processed Data**:
  - Saves the processed data as a compressed **.npz** file (`hand_dataset.npz`).

### Example of Usage:
```bash
python preprocessing.py
```
This command processes the images and prepares them for model training.

---

## 3. `train_model.py` - CNN Model Definition and Training

This script defines and trains a **Convolutional Neural Network (CNN)** to classify hand gestures.

### Key Steps:

- **Loading Data**:
  - Loads the preprocessed dataset (`hand_dataset.npz`).

- **Data Augmentation**:
  - Applies real-time augmentation using **ImageDataGenerator**.

- **Model Architecture**:
  - Constructs a CNN with the following layers:
    - **Conv2D**: Extracts features from images.
    - **MaxPooling2D**: Reduces dimensionality.
    - **Flatten**: Converts features to a 1D vector.
    - **Dense**: Fully connected layers for classification.
    - **Dropout**: Helps prevent overfitting.
    - **Softmax**: Outputs probabilities for each class.

- **Model Compilation**:
  - Compiles the model using the **Adam optimizer** and **categorical crossentropy loss**.

- **Training**:
  - Trains the model for **35 epochs** and validates its performance.

- **Saving Model**:
  - Saves the trained model as `cnn_model.h5`.

### Example of Usage:
```bash
python train_model.py
```
This command trains the CNN and saves the model for later use.

---

## 4. `sign_detector.py` - Real-Time Gesture Recognition and Wheelchair Control

This is the main script that integrates gesture recognition and motor control, continuously processing webcam images.

### Key Steps:

- **Loading the Trained Model**:
  - Loads the saved model (`cnn_model.h5`) for gesture prediction.

- **Real-Time Webcam Feed**:
  - Captures images from the webcam continuously.

- **Gesture Prediction**:
  - Preprocesses each captured frame and feeds it into the CNN.
  - Predicts the gesture and determines the corresponding action (e.g., forward, backward).

- **Motor Control Logic**:
  - Sends commands to the motor driver (L298) to control wheelchair movements:
    - **Forward**, **Backward**, **Left**, **Right**, **Stop**.

- **Feedback Mechanisms**:
  - Uses LEDs and buzzers for feedback:
    - **Green LED**: Successful gesture recognition.
    - **Blue LED**: Incorrect gesture detected.
    - **Red LED**: Stop movement.

- **Obstacle Detection**:
  - Utilizes the **HC-SR04 distance sensor** to detect obstacles.
  - Stops the wheelchair if an object is detected within 15 cm.

### Example of Usage:
```bash
python sign_detector.py
```
This command starts the system, capturing real-time gestures and controlling the wheelchair based on user input.

---
 - by Aditya Kumar