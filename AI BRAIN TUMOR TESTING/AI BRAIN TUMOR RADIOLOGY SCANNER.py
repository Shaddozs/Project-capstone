import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load your trained multi-class model
model_path = 'E:\AI BRAIN TUMOR TESTING\PRE TRAINED MODELS\RESNET 50 MODEL.h5'
model = keras.models.load_model(model_path)


# Function to preprocess a single image
def preprocess_image(image_path, target_size=(200, 200)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    img = cv2.resize(img, target_size)  # Resize to the model's input size
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Path to the radiology image you want to classify
image_path = 'E:\AI BRAIN TUMOR TESTING\IMAGES\TESTING MENINGIOMA.jpg'

print(f"Loading image from: E:\AI BRAIN TUMOR TESTING\IMAGES\TESTING MENINGIOMA.jpg")
img = cv2.imread(image_path)

# Preprocess the image
processed_image = preprocess_image(image_path)

# Perform inference
predictions = model.predict(np.expand_dims(processed_image, axis=0))

# Interpret the predictions
class_indices = np.argmax(predictions, axis=1)
class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']  # Replace with your class names

predicted_tumor_type = class_names[class_indices[0]]

print(f"The image is classified as: {predicted_tumor_type}")
