import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('models/object_detection_model.h5')

def detect_objects(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))  # Resize to the model input size
    image_resized = image_resized / 255.0  # Normalize to [0, 1]
    image_input = np.expand_dims(image_resized, axis=0)

    predictions = model.predict(image_input)
    class_id = np.argmax(predictions)
    
    # Here we just show the class ID, but you would typically also handle bounding boxes
    print(f"Predicted Class ID: {class_id}")
    cv2.imshow('Detected Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run inference on a test image
detect_objects('data/images/sample_image.jpg')
