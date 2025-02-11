import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    # Load dataset
    image_paths = [os.path.join('data/images', f) for f in os.listdir('data/images')]
    labels = [0] * len(image_paths)  # Placeholder, replace with actual labels
    
    # Train/test split
    train_images, val_images, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2)
    
    # Create TensorFlow dataset
    def load_image(image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Resize to (224, 224)
        image = image / 255.0  # Normalize
        return image

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.map(lambda x, y: (load_image(x), y))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.map(lambda x, y: (load_image(x), y))
    val_dataset = val_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset
