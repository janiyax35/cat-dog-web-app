from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("cat_dog_model.h5")

UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded"
    
    file = request.files['image']
    if file.filename == '':
        return "No file selected"
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=(64, 64))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0

    # Prediction
    prediction = model.predict(img_tensor)[0][0]
    result = "It's a Dog 🐶" if prediction > 0.5 else "It's a Cat 🐱"
    return render_template('index.html', result=result, image_url=filepath)

if __name__ == '__main__':
    app.run(debug=True)
