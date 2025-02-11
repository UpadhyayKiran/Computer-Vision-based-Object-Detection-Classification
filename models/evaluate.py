import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import cv2
from models.utils import preprocess_image, load_annotations

# Load the test annotations and the trained model
def load_model_and_annotations(model_path, annotation_file, image_dir):
    # Load the trained model
    model = load_model(model_path)

    # Load the test annotations
    with open(annotation_file, 'r') as file:
        annotations = json.load(file)

    return model, annotations

# Evaluate the model on the test set
def evaluate_model(model, annotations, image_dir):
    # Store results
    true_labels = []
    predicted_labels = []
    confidence_scores = []

    for annotation in annotations['images']:
        image_id = annotation['id']
        image_path = os.path.join(image_dir, annotation['file_name'])

        # Preprocess the image (resize, normalize, etc.)
        image = preprocess_image(image_path)

        # Get true labels from annotation
        true_labels.append([obj['category_id'] for obj in annotation['objects']])

        # Predict using the trained model
        image_input = np.expand_dims(image, axis=0)
        predictions = model.predict(image_input)

        # Assuming the model returns [boxes, class_probs, scores]
        boxes, class_probs, scores = predictions

        # Find the predicted class based on the highest score
        predicted_class = np.argmax(class_probs, axis=1)
        predicted_labels.append(predicted_class)

        # Store confidence scores
        confidence_scores.append(np.max(scores))

    return true_labels, predicted_labels, confidence_scores

# Compute mean Average Precision (mAP)
def compute_mAP(true_labels, predicted_labels, confidence_scores, num_classes):
    # Flatten the lists
    true_labels = np.array(true_labels).flatten()
    predicted_labels = np.array(predicted_labels).flatten()
    confidence_scores = np.array(confidence_scores).flatten()

    # Compute average precision per class
    average_precision = []
    for class_id in range(num_classes):
        # Extract true/false positives and confidence scores for this class
        class_true = (true_labels == class_id).astype(int)
        class_pred = (predicted_labels == class_id).astype(int)

        # Compute precision-recall curve
        ap = average_precision_score(class_true, confidence_scores)
        average_precision.append(ap)

    # Compute mAP (mean of average precisions)
    mAP = np.mean(average_precision)
    return mAP

# Plot the evaluation results
def plot_results(true_labels, predicted_labels, confidence_scores):
    # Example of plotting a confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def main():
    # File paths
    model_path = 'models/object_detection_model.h5'
    annotation_file = 'data/annotations.json'
    image_dir = 'data/images/'

    # Number of classes in your dataset (including background)
    num_classes = 10  # Adjust this based on your dataset

    # Load model and annotations
    model, annotations = load_model_and_annotations(model_path, annotation_file, image_dir)

    # Evaluate the model on the test data
    true_labels, predicted_labels, confidence_scores = evaluate_model(model, annotations, image_dir)

    # Compute mAP
    mAP = compute_mAP(true_labels, predicted_labels, confidence_scores, num_classes)
    print(f"Mean Average Precision (mAP): {mAP:.4f}")

    # Plot results (confusion matrix, precision-recall, etc.)
    plot_results(true_labels, predicted_labels, confidence_scores)

if __name__ == "__main__":
    main()
