# **Smart Wheelchair Using CNN**

#### **1. Overview**

This project presents the design and implementation of a smart wheelchair that utilizes Convolutional Neural Networks (CNN) to recognize specific hand gestures and translate them into directional controls. The system allows users to control the wheelchair’s movements by simply performing predefined hand gestures. The core system components include a camera for capturing hand images, a CNN model for gesture recognition, a Raspberry Pi as the control unit, an L298 motor driver, and four gear motors to drive the wheelchair.

#### **2. System Architecture**

The smart wheelchair system is comprised of both software and hardware components.

**Software Component:**
- **Hand Gesture Recognition Using CNN**:
  - The software component involves the development of a CNN model that is trained to recognize six specific hand gestures. Each gesture corresponds to a specific movement command for the wheelchair:
    - **Palm**: Stop the wheelchair.
    - **Index Finger + Pinky**: Move forward.
    - **Middle Finger + Ring Finger**: Move backward.
    - **Index Finger**: Turn left.
    - **Pinky Finger**: Turn right.
    - **Fist**: Stop the wheelchair.

**Hardware Component:**
- **Raspberry Pi**:
  - The Raspberry Pi acts as the central processing unit, processing the images captured by the camera, running the CNN model, and sending movement commands to the wheelchair’s motors based on the recognized gestures.

- **Camera**:
  - The camera captures real-time images of the user’s hand and feeds these images to the Raspberry Pi for processing.

- **L298 Motor Driver**:
  - The L298 motor driver is responsible for translating the Raspberry Pi’s commands into signals that control the wheelchair's motors.

- **Gear Motors**:
  - The four gear motors, connected to the wheels of the wheelchair, execute the movements such as forward, backward, left, right, and stop based on the signals received from the L298 motor driver.

#### **3. Hand Gesture Recognition Using CNN**

The hand gesture recognition system is the core of the wheelchair's control mechanism, built using a CNN trained on a dataset of hand gestures.

- **Data Collection**:
  - A dataset of hand images is collected, each labeled with the corresponding gesture: palm, index + pinky, middle + ring, index, pinky, and fist. The dataset is diversified with images under various lighting conditions and hand orientations to ensure robust gesture recognition.

- **Data Preprocessing**:
  - The images are preprocessed by resizing, normalizing, and possibly augmenting them to improve the CNN model’s performance. This preprocessing ensures the model can generalize well to different users and environments.

- **CNN Model Architecture**:
  - The CNN model is designed with multiple convolutional and pooling layers to extract features from the hand images, followed by fully connected layers that classify the gestures. The final output layer maps these features to one of the six possible gestures.

- **Model Training**:
  - The model is trained using a supervised learning approach. The preprocessed dataset is divided into training and validation sets. During training, the CNN learns to recognize patterns associated with each gesture by minimizing a loss function that measures the difference between predicted and actual labels.

- **Gesture Recognition**:
  - Once trained, the CNN model is deployed on the Raspberry Pi. The camera captures hand images, which are processed in real-time by the CNN to predict the gesture. Based on the predicted gesture, the Raspberry Pi sends the corresponding control command to the wheelchair.

#### **4. Hardware Integration**

The recognized hand gestures are converted into movement commands that control the wheelchair through the Raspberry Pi, L298 motor driver, and gear motors.

- **Raspberry Pi**:
  - The Raspberry Pi is programmed to handle real-time image processing and motor control. It interfaces with the camera to capture images, runs the CNN model to identify the gesture, and sends control signals to the L298 motor driver to move the wheelchair accordingly.

- **L298 Motor Driver**:
  - The L298 motor driver interfaces between the Raspberry Pi and the gear motors. It receives low-power signals from the Raspberry Pi and converts them into high-power signals required to drive the motors. The driver controls the direction and speed of the motors, enabling the wheelchair to perform the desired movements.

- **Gear Motors**:
  - The four gear motors control the wheelchair's wheels. Their coordinated operation determines the wheelchair's movement direction:
    - **Move Forward**: Both rear motors move forward when the "Index + Pinky" gesture is recognized.
    - **Move Backward**: Both rear motors move backward when the "Middle + Ring" gesture is recognized.
    - **Turn Left**: The right motor moves forward while the left motor stops or moves backward when the "Index Finger" gesture is recognized.
    - **Turn Right**: The left motor moves forward while the right motor stops or moves backward when the "Pinky Finger" gesture is recognized.
    - **Stop**: All motors stop when the "Palm" or "Fist" gesture is recognized.

#### **5. Gesture to Movement Mapping**

Each hand gesture is mapped to a specific wheelchair movement:

- **Palm (Stop)**: The wheelchair stops moving.
- **Index Finger + Pinky (Move Forward)**: The wheelchair moves forward.
- **Middle Finger + Ring Finger (Move Backward)**: The wheelchair moves backward.
- **Index Finger (Turn Left)**: The wheelchair turns left.
- **Pinky Finger (Turn Right)**: The wheelchair turns right.
- **Fist (Stop)**: The wheelchair stops moving.

#### **6. Testing and Validation**

The system is tested to ensure that it accurately recognizes gestures and translates them into the correct wheelchair movements.

- **Gesture Recognition Accuracy**:
  - The CNN model's accuracy in recognizing gestures is tested across different users, hand orientations, and lighting conditions. The goal is to achieve a high recognition rate with minimal false positives.

- **Response Time**:
  - The time taken from gesture recognition to wheelchair movement is measured to ensure that the system operates in real-time, providing immediate feedback to the user.

- **Reliability**:
  - The system is evaluated for reliability, ensuring consistent operation over extended periods without failure.

#### **7. Conclusion**

The smart wheelchair system provides an innovative and intuitive solution for individuals with mobility impairments. By leveraging CNNs for gesture recognition, the wheelchair can be controlled using simple hand gestures, making it accessible and easy to use. This project demonstrates the potential of integrating AI with assistive technologies to enhance the independence and quality of life for users.

Further improvements could include expanding the range of gestures, optimizing the CNN model for even faster processing, and integrating additional safety features to ensure the system's robustness in various real-world scenarios.

- by Aditya Kumar