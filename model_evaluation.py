import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define output directory
output_dir = 'output/model'
os.makedirs(output_dir, exist_ok=True)

# Load dataset
data = np.load('hand_dataset.npz')
train_images = data['train_images']
train_labels = data['train_labels']
val_images = data['val_images']
val_labels = data['val_labels']
test_images = data['test_images']
test_labels = data['test_labels']
class_names = data['class_names']

# Convert labels to one-hot encoding
num_classes = len(class_names)
train_labels_categorical = to_categorical(train_labels, num_classes)
val_labels_categorical = to_categorical(val_labels, num_classes)
test_labels_categorical = to_categorical(test_labels, num_classes)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, zoom_range=0.1, horizontal_flip=True
)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_images, train_labels_categorical, batch_size=32)
val_generator = val_datagen.flow(val_images, val_labels_categorical, batch_size=32)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(train_images.shape[1], train_images.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=35, validation_data=val_generator)

# Evaluate on test data
loss, accuracy = model.evaluate(test_images, test_labels_categorical, verbose=0)

# Print accuracy and loss
print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")
with open(os.path.join(output_dir, 'performance.txt'), 'w') as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\nTest Loss: {loss:.4f}\n")

# Plot accuracy and loss over epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
plt.close()

# Confusion Matrix
test_predictions = model.predict(test_images)
test_pred_labels = np.argmax(test_predictions, axis=1)

conf_matrix = confusion_matrix(test_labels, test_pred_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# Classification Report
report = classification_report(test_labels, test_pred_labels, target_names=class_names)

# Save classification report
with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Accuracy Bar Chart
accuracy_values = np.mean(test_predictions, axis=0)
plt.figure(figsize=(10, 6))
plt.bar(class_names, accuracy_values, color='lightblue')
plt.title('Class-wise Accuracy')
plt.xlabel('Classes')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig(os.path.join(output_dir, 'class_accuracy_bar_chart.png'))
plt.close()

# Generate a Gantt-like chart for training and validation process visualization (custom timeline view)
epochs = list(range(1, len(history.history['accuracy']) + 1))
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label='Train Accuracy', color='blue', marker='o')
plt.plot(epochs, val_acc, label='Validation Accuracy', color='green', marker='o')
plt.plot(epochs, train_loss, label='Train Loss', color='red', linestyle='--', marker='x')
plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linestyle='--', marker='x')
plt.title('Training & Validation Progress (Custom View)')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'gantt_like_chart.png'))
plt.close()

# Log additional important parameters
with open(os.path.join(output_dir, 'training_parameters.txt'), 'w') as f:
    f.write("Training Parameters:\n")
    f.write(f"Number of Classes: {num_classes}\n")
    f.write(f"Number of Epochs: {35}\n")
    f.write(f"Batch Size: {32}\n")
    f.write(f"Test Loss: {loss:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")

print(f"All outputs saved in {output_dir}")
