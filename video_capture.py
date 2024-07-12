import cv2
import numpy as np
from keras.models import load_model

# Ruta del modelo encoder preentrenado
encoder_model_path = 'encoder_model.h5'
encoder = load_model(encoder_model_path)

# Función para preprocesar el iris
def preprocess_iris(image, target_size=(224, 224)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, target_size)
    normalized = resized / 255.0
    return normalized

# Función para calcular la nitidez de una imagen
def calculate_sharpness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Función para detectar el iris en un fotograma
def detect_iris(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity > 0.7 and area > 100:
            return True
    return False

# Función principal para capturar y procesar imágenes de video
def capture_iris_images():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    sharp_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        preprocessed_frame = preprocess_iris(frame)
        sharpness = calculate_sharpness(preprocessed_frame)

        if sharpness > 100 and detect_iris(frame):
            sharp_frames.append(preprocessed_frame)
            if len(sharp_frames) >= 5:
                break

    cap.release()
    cv2.destroyAllWindows()
    
    if len(sharp_frames) < 5:
        raise ValueError("No se pudieron capturar suficientes imágenes de iris nítidas.")
    
    sharp_frames = np.array(sharp_frames)
    embeddings = encoder.predict(sharp_frames)
    return embeddings
