import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    img = cv2.resize(img, (28, 28))  # Resize image to 28x28 pixels
    img = img.astype('float32') / 255.0  # Normalize pixel values to [0, 1] 
    img = np.expand_dims(img, axis=(0, -1))  # Add batch and channel dimensions
    return img

model = load_model("D:/malo/Documents/cours_tsp/cv/TP2_MNIST_Fashion_Classification_Moodle/TP2_MNIST_Fashion_Classification_Moodle/Fashion_MNIST_model.h5")

test_image = load_image("D:/malo/Documents/cours_tsp/cv/TP2_MNIST_Fashion_Classification_Moodle/TP2_MNIST_Fashion_Classification_Moodle/sample_image.png")

prediction = model.predict(test_image)
predicted_class = np.argmax(prediction)

print(f"Predicted class: {predicted_class}")
