import os
import json
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

# Function to load the dataset annotations
def load_annotations(annotation_file):
    with open(annotation_file, 'r') as file:
        annotations = json.load(file)
    return annotations

# Function to preprocess individual images
def preprocess_image(image_path, target_size=(224, 224)):
    # Read the image
    img = cv2.imread(image_path)
    # Resize the image to the target size
    img_resized = cv2.resize(img, target_size)
    # Normalize the image pixel values to [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0
    return img_normalized

# Function to preprocess data
def preprocess_data(annotations, image_dir, target_size=(224, 224)):
    images = []
    boxes = []
    labels = []

    # Iterate through each image annotation
    for annotation in annotations['images']:
        image_path = os.path.join(image_dir, annotation['file_name'])
        
        # Preprocess the image (resize, normalize)
        image_data = preprocess_image(image_path, target_size)
        images.append(image_data)

        # Prepare the bounding boxes and labels
        image_boxes = []
        image_labels = []
        for obj in annotation['objects']:
            # Get the bounding box coordinates (x, y, width, height)
            bbox = obj['bbox']
            image_boxes.append(bbox)

            # Get the category (label)
            image_labels.append(obj['category_id'])

        boxes.append(image_boxes)
        labels.append(image_labels)

    # Convert lists to numpy arrays
    images = np.array(images)
    boxes = np.array(boxes)
    labels = np.array(labels)

    return images, boxes, labels

# Example usage of the preprocessing function
def main():
    annotation_file = 'data/annotations.json'
    image_dir = 'data/images/'
    
    # Load annotations
    annotations = load_annotations(annotation_file)
    
    # Preprocess data (images, bounding boxes, and labels)
    images, boxes, labels = preprocess_data(annotations, image_dir)

    # Save the processed data (if needed for future use)
    np.save('data/images.npy', images)
    np.save('data/boxes.npy', boxes)
    np.save('data/labels.npy', labels)

    print(f"Processed {len(images)} images and saved to data/images.npy")

if __name__ == "__main__":
    main()
