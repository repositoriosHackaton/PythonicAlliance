document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureButton = document.getElementById('capture-button');
    const registerButton = document.getElementById('register-button');
    const verifyButton = document.getElementById('verify-button');
    const resultsDiv = document.getElementById('results');

    // Solicitar acceso a la cámara del usuario
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
            video.srcObject = stream;
            video.play();
        });
    }

    // Capturar un frame de video cuando se hace clic en el botón
    captureButton.addEventListener('click', function() {
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convertir la imagen del canvas a base64
        const imageData = canvas.toDataURL('image/png');

        // Mostrar la imagen capturada
        const img = document.createElement('img');
        img.src = imageData;
        resultsDiv.appendChild(img);
    });

    // Enviar imágenes capturadas para el registro
    if (registerButton) {
        registerButton.addEventListener('click', function() {
            const images = [];
            document.querySelectorAll('#results img').forEach(img => {
                images.push(img.src);
            });
            fetch('/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ images })
            }).then(response => response.json())
              .then(data => alert(data.message));
        });
    }

    // Enviar imagen capturada para la verificación
    if (verifyButton) {
        verifyButton.addEventListener('click', function() {
            const images = [];
            document.querySelectorAll('#results img').forEach(img => {
                images.push(img.src);
            });
            fetch('/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: images[0] })
            }).then(response => response.json())
              .then(data => alert(data.message));
        });
    }
});
