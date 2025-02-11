import tensorflow as tf
from models.model import build_model
from models.utils import load_data
import os

# Load data
train_dataset, val_dataset = load_data()

# Build model
model = build_model(input_shape=(224, 224, 3), num_classes=80)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    steps_per_epoch=len(train_dataset),
    validation_steps=len(val_dataset)
)

# Save the trained model
model.save('models/object_detection_model.h5')

# Save plots
import matplotlib.pyplot as plt

# Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'])
plt.savefig('outputs/graphs/loss_plot.png')

# Accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'])
plt.savefig('outputs/graphs/accuracy_plot.png')
