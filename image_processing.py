import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_iris(image, target_size=(224, 224)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, target_size)
    normalized = resized / 255.0
    return normalized

def generate_embedding(model, image):
    image = np.expand_dims(image, axis=0)
    embedding = model.predict(image)
    return embedding.flatten()

def handle_reflection(image, threshold=240):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    reflection_mask = gray > threshold
    kernel = np.ones((5, 5), np.uint8)
    reflection_mask = cv2.dilate(reflection_mask.astype(np.uint8), kernel, iterations=1)
    result = cv2.inpaint(image, reflection_mask, 3, cv2.INPAINT_TELEA)
    return result

def check_image_quality(image, min_sharpness=100, min_brightness=50, max_brightness=220):
    image = handle_reflection(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    is_sharp = laplacian_var > min_sharpness
    is_bright_enough = min_brightness < brightness < max_brightness
    eye_detected = len(eyes) > 0
    return is_sharp and is_bright_enough and eye_detected
