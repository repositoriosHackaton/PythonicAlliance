from flask import Blueprint, request, jsonify, render_template
import cv2
import numpy as np
import base64
import os
from app.utils import preprocess_iris, generate_embedding, handle_reflection, check_image_quality
from iris_model import load_encoder
from database import save_user_embeddings, find_best_match
from video_capture import capture_iris_images

main = Blueprint('main', __name__)

encoder = load_encoder()

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        try:
            embeddings = capture_iris_images()
            save_user_embeddings(username, embeddings)
            return jsonify({'message': 'Registro exitoso'})
        except Exception as e:
            return jsonify({'message': str(e)}), 400

    return render_template('register.html')

@main.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        username = request.form['username']
        iris_image = request.form['iris_image']
        nparr = np.frombuffer(base64.b64decode(iris_image.split(',')[1]), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if not check_image_quality(image):
            return jsonify({'message': 'La calidad de la imagen no es suficiente'}), 400

        processed_image = preprocess_iris(image)
        new_embedding = generate_embedding(encoder, processed_image)
        user, similarity = find_best_match(new_embedding)

        if user == username and similarity > 0.85:
            return jsonify({'message': 'Acceso permitido', 'user': user})
        else:
            return jsonify({'message': 'Acceso denegado'}), 401

    return render_template('verify.html')

if __name__ == '__main__':
    main.run(debug=True)