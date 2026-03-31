from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load your model
model = load_model('models/sign_model1.h5')

# IMPORTANT: match your dataset folder names exactly
classes = [
    "Cool", "Good", "Good luck", "hello", "i love you",
    "Its OK", "little bit", "Loser", "Peace", "Power"
]

def preprocess_image(image_path):
    img = Image.open(image_path).resize((64, 64))  # ⚠️ match your training size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        file = request.files['file']

        if file:
            filepath = os.path.join('static/uploads', file.filename)
            file.save(filepath)

            img = preprocess_image(filepath)
            pred = model.predict(img)

            prediction = classes[np.argmax(pred)]
            confidence = float(np.max(pred)) * 100
            image_path = filepath

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == '__main__':
    app.run(debug=True)