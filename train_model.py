import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from tensorflow.keras.callbacks import TensorBoard
import pickle


dataset_path = r'Proyecto FINAL IA OcularShield\preprocessed_iris_images.pkl'

with open(dataset_path, 'rb') as f:
    images = pickle.load(f)

images = np.expand_dims(images, axis=-1)

input_img = Input(shape=(224, 224, 1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = Dropout(0.5)(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Dropout(0.5)(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = Dropout(0.5)(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Dropout(0.5)(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Dropout(0.5)(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Definir la ruta del directorio del proyecto
project_dir = r'Proyecto FINAL IA OcularShield'

# Crear la ruta completa para la carpeta 'logs'
log_dir = os.path.join(project_dir, 'logs')

tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

autoencoder.fit(images, images, epochs=200, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[tensorboard])

encoder = Model(input_img, encoded)
encoder.save('encoder_model.h5')

encoder_loaded = load_model('encoder_model.h5')

encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
