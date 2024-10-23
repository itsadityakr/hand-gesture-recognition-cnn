import os
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the dataset from the saved .npz file
data = np.load('hand_dataset.npz')

# Extract the training, validation, and test datasets
train_images = data['train_images']
train_labels = data['train_labels']
val_images = data['val_images']
val_labels = data['val_labels']
test_images = data['test_images']
test_labels = data['test_labels']
class_names = data['class_names']

# Create output directory
output_dir = '.'
os.makedirs(output_dir, exist_ok=True)

# Convert labels to categorical (one-hot encoding)
num_classes = len(class_names)
train_labels_categorical = to_categorical(train_labels, num_classes)
val_labels_categorical = to_categorical(val_labels, num_classes)
test_labels_categorical = to_categorical(test_labels, num_classes)

# Define the image size
image_size = train_images.shape[1]

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_images, train_labels_categorical, batch_size=32)
val_generator = val_datagen.flow(val_images, val_labels_categorical, batch_size=32)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=35, validation_data=val_generator)

# Evaluate the model on the test data
loss, acc = model.evaluate(test_images, test_labels_categorical, verbose=0)

# Save the model
model.save(os.path.join(output_dir, 'cnn_model.h5'))

# Inform the user
print(f'Model saved as cnn_model.h5 in {output_dir}')