import tensorflow as tf
from tensorflow.keras.models import load_model

def load_encoder():
    try:
        encoder_loaded = load_model('encoder_model.h5')
        encoder_loaded.summary()
        print("El modelo se cargó correctamente. No está corrupto.")
        return encoder_loaded
    except Exception as e:
        print(f"Ocurrió un error al intentar cargar el modelo: {e}")
        raise
