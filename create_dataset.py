import os
import cv2
import numpy as np
import pickle

def preprocess_iris(image, target_size=(224, 224)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, target_size)
    normalized = resized / 255.0
    return normalized

# Ruta de la carpeta con las imágenes
src_folder = r'DATASET_FULL_IMAGES'

# Ruta de la carpeta del proyecto
project_folder = r'Proyecto FINAL IA OcularShield'

# Ruta completa donde se guardará el archivo
output_file = os.path.join(project_folder, 'preprocessed_iris_images.pkl')

# Crear una lista para almacenar las imágenes preprocesadas
images = []

# Procesar y almacenar las imágenes
for filename in os.listdir(src_folder):
    filepath = os.path.join(src_folder, filename)
    if os.path.isfile(filepath):
        image = cv2.imread(filepath)
        if image is not None:
            processed_image = preprocess_iris(image)
            if processed_image is not None:
                images.append(processed_image)
            else:
                print(f'No se pudo procesar {filename}')
        else:
            print(f'No se pudo leer la imagen {filename}')

# Convertir la lista a un array de NumPy
images = np.array(images)

# Guardar el array en un archivo en la carpeta del proyecto
with open(output_file, 'wb') as f:
    pickle.dump(images, f)

print(f'Dataset preprocesado y guardado exitosamente en {output_file}. Total de imágenes procesadas: {len(images)}')