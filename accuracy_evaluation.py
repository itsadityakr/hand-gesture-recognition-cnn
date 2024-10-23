import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the size to which the hand image will be resized for the CNN model
image_size = 32

# Load the trained ASL recognition model
model = load_model('cnn_model.h5')

# Initialize MediaPipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize variables for metrics
y_true_total = []  # Store all true labels
y_pred_total = []  # Store all predicted labels
metrics_per_letter = {}  # Dictionary to store metrics for each letter
asl_letters = 'BFLRS'

# Create output directory for evaluation
evaluation_dir = 'output/accuracy'
os.makedirs(evaluation_dir, exist_ok=True)

# Define the function to preprocess the hand image before sending it to the CNN model
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.flip(gray, 1)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresholded = cv2.threshold(gray, 161, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(thresholded, (image_size, image_size))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, image_size, image_size, 1))
    return reshaped, thresholded

# Define a function to map the CNN model's prediction to the corresponding ASL letter
def predict_asl_letter(prediction):
    asl_letters = 'BFLRS'
    return asl_letters[prediction]

# Function to run predictions for a specified letter for 5 seconds
def run_predictions(true_label):
    y_true = []  # Clear the true labels for each letter
    y_pred = []  # Clear the predicted labels for each letter

    start_time = time.time()
    while time.time() - start_time < 5:  # Adjust to 5 seconds
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                h, w, c = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                margin = 30
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)

                hand_image = frame[y_min:y_max, x_min:x_max]
                preprocessed_image, resized_image = preprocess_image(hand_image)

                prediction = model.predict(preprocessed_image)
                predicted_label = np.argmax(prediction)
                confidence = np.max(prediction) * 100  # Calculate the confidence of the prediction

                asl_letter = predict_asl_letter(predicted_label)

                # Store true and predicted labels for metrics
                y_true.append(true_label)
                y_pred.append(asl_letter)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f'ASL Letter: {asl_letter} ({confidence:.2f}%)', 
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Preprocessed Image', resized_image)

        cv2.imshow('ASL Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    return y_true, y_pred  # Return the true and predicted labels for this session

# Start video capture
cap = cv2.VideoCapture(0)

# Open a text file to save terminal output
with open(os.path.join(evaluation_dir, 'evaluation_output.txt'), 'w') as f:
    for true_label in asl_letters:
        print(f"Starting prediction for letter: {true_label}")
        f.write(f"Starting prediction for letter: {true_label}\n")

        print("Press 'C' to start prediction for 5 seconds or 'Q' to quit.")
        f.write("Press 'C' to start prediction for 5 seconds or 'Q' to quit.\n")

        while True:
            ret, frame = cap.read()
            cv2.imshow('ASL Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                print(f"Starting prediction for {true_label} for 5 seconds...")
                f.write(f"Starting prediction for {true_label} for 5 seconds...\n")
                
                y_true, y_pred = run_predictions(true_label)

                # Calculate metrics after prediction session
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

                # Calculate the most common predicted label
                if y_pred:
                    most_common_label, count = Counter(y_pred).most_common(1)[0]
                else:
                    most_common_label = "None"
                    count = 0

                letter_output = (f'Letter: {true_label} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, '
                                 f'Recall: {recall:.2f}, F1 Score: {f1:.2f}')
                print(letter_output)
                f.write(letter_output + '\n')  # Write to the output file

                # Store total metrics for this letter
                metrics_per_letter[true_label] = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'Most Predicted Label': most_common_label,
                    'Count': count
                }

                # Store total metrics
                y_true_total.extend([true_label] * len(y_pred))
                y_pred_total.extend(y_pred)

                break  # Break out of the inner loop to proceed to the next letter

            elif key == ord('q'):
                break

# Calculate overall metrics
if y_true_total:
    overall_accuracy = accuracy_score(y_true_total, y_pred_total)
    overall_precision = precision_score(y_true_total, y_pred_total, average='weighted', zero_division=0)
    overall_recall = recall_score(y_true_total, y_pred_total, average='weighted', zero_division=0)
    overall_f1 = f1_score(y_true_total, y_pred_total, average='weighted', zero_division=0)

    overall_output = (f'\nOverall Metrics:\n'
                      f'Overall Accuracy: {overall_accuracy:.2f}, Overall Precision: {overall_precision:.2f}, '
                      f'Overall Recall: {overall_recall:.2f}, Overall F1 Score: {overall_f1:.2f}')

    # Make sure the file is still open before writing
    with open(os.path.join(evaluation_dir, 'evaluation_output.txt'), 'a') as f:
        f.write(overall_output + '\n')  # Write to the output file

    print(overall_output)

    print("\nMetrics for Each Letter:")
    with open(os.path.join(evaluation_dir, 'evaluation_output.txt'), 'a') as f:
        f.write("\nMetrics for Each Letter:\n")  # Write header to the output file
        for letter, metrics in metrics_per_letter.items():
            letter_output = (f'Letter: {letter} - Accuracy: {metrics["Accuracy"]:.2f}, '
                             f'Precision: {metrics["Precision"]:.2f}, Recall: {metrics["Recall"]:.2f}, '
                             f'F1 Score: {metrics["F1 Score"]:.2f}, Most Predicted Label: {metrics["Most Predicted Label"]} (Count: {metrics["Count"]})')
            print(letter_output)
            f.write(letter_output + '\n')  # Write to the output file

    # Create confusion matrix
    cm = confusion_matrix(y_true_total, y_pred_total, labels=list(asl_letters))

    # Save confusion matrix as a text file
    confusion_matrix_file = os.path.join(evaluation_dir, 'confusion_matrix.txt')
    with open(confusion_matrix_file, 'w') as f:
        f.write('Confusion Matrix:\n')
        f.write('True\\Predicted\t' + '\t'.join(list(asl_letters)) + '\n')  # Header
        for i, row in enumerate(cm):
            f.write(f'{list(asl_letters)[i]}\t' + '\t'.join(map(str, row)) + '\n')

    # Now plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Use light blue color for confusion matrix
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(ticks=np.arange(len(asl_letters)) + 0.5, labels=list(asl_letters), rotation=45)
    plt.yticks(ticks=np.arange(len(asl_letters)) + 0.5, labels=list(asl_letters), rotation=0)
    plt.savefig(os.path.join(evaluation_dir, 'confusion_matrix.png'))  # Save confusion matrix plot
    plt.close()  # Close the plot to free memory

    # Create colorful bar plots for metrics
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for metric in metrics_list:
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_per_letter.keys(), [metrics_per_letter[letter][metric] for letter in asl_letters],
                color=sns.color_palette("husl", len(asl_letters)))  # Colorful bar plot
        plt.title(f'{metric} per Letter', fontsize=16)
        plt.xlabel('ASL Letter', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
        plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline')  # Add a baseline
        plt.legend()
        plt.savefig(os.path.join(evaluation_dir, f'{metric.lower()}_per_letter.png'))  # Save metrics plot
        plt.close()  # Close the plot to free memory

else:
    print("\nNo predictions were made.")
    with open(os.path.join(evaluation_dir, 'evaluation_output.txt'), 'a') as f:
        f.write("No predictions were made.\n")  # Write to the output file

cap.release()
cv2.destroyAllWindows()