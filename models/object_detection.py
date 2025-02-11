import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

# Define paths
IMAGE_PATH = 'data/images/'
ANNOTATIONS_PATH = 'data/annotations.json'
MODEL_SAVE_PATH = 'models/object_detection_model.h5'

# Load annotations from JSON (assuming COCO-style format or similar)
def load_annotations():
    with open(ANNOTATIONS_PATH, 'r') as f:
        annotations = json.load(f)
    return annotations

# Load and preprocess the images and annotations
def load_data(annotations):
    images = []
    labels = []
    
    for annotation in tqdm(annotations['images']):
        image_path = os.path.join(IMAGE_PATH, annotation['file_name'])
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))  # Resize to match model input
        
        # Normalize the image
        img = img.astype('float32') / 255.0
        
        # Here we assume that 'annotations' contain bounding boxes and class ids for each image
        # You should adjust this part based on the exact structure of your annotations
        image_labels = []
        for obj in annotations['annotations']:
            if obj['image_id'] == annotation['id']:
                image_labels.append(obj)
        
        images.append(img)
        labels.append(image_labels)
    
    return np.array(images), labels

# Define a simple CNN model architecture
def create_model(input_shape=(224, 224, 3)):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2))  # Assuming binary classification or one object per image (adjust if needed)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['accuracy'])
    return model

# Function to train the model
def train_model():
    annotations = load_annotations()  # Load annotations from JSON
    images, labels = load_data(annotations)  # Load and preprocess data

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Create the model
    model = create_model()

    # Train the model
    model.fit(X_train, np.array(y_train), epochs=10, batch_size=32, validation_data=(X_val, np.array(y_val)))

    # Save the trained model
    model.save(MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')

if __name__ == '__main__':
    train_model()
