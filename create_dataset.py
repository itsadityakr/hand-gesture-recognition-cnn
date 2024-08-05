import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Suppress TensorFlow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess_image(image_path, target_size=(64, 64)):
    image = load_img(image_path, target_size=target_size, color_mode='grayscale')
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # Normalize to [0, 1]
    return image_array

def create_dataset(directory, target_size=(64, 64)):
    images = []
    labels = []
    for file in os.listdir(directory):
        image_path = os.path.join(directory, file)
        # Extract label from filename (assuming format like '1_0.jpg')
        label = file.split('_')[0]
        if label.isdigit() and int(label) in range(1, 7):  # Check if the label is from 1 to 6
            images.append(preprocess_image(image_path, target_size))
            labels.append(label)
    return np.array(images), np.array(labels)
