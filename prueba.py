import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    # Intenta cargar el modelo
    encoder_loaded = load_model('Proyecto FINAL IA OcularShield\encoder_model.h5')
    encoder_loaded.summary()
    print("El modelo se cargó correctamente. No está corrupto.")
except Exception as e:
    print(f"Ocurrió un error al intentar cargar el modelo: {e}")
