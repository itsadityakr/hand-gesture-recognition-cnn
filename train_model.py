import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from create_dataset import create_dataset

# Define the class labels
class_labels = ['1', '2', '3', '4', '5', '6']

# Load dataset
images, labels = create_dataset('dataset/')
images = images.reshape(-1, 64, 64, 1)  # Reshape for CNN

# Convert labels to numeric values
label_to_index = {label: index for index, label in enumerate(class_labels)}
numeric_labels = np.array([label_to_index[label] for label in labels if label in label_to_index])

# Create and train model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(class_labels), activation='softmax')  # Number of classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(images, numeric_labels, epochs=10, validation_split=0.2)

# Save the model
model.save('hand_gesture_model.h5')
