import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("model.h5")
with open("class_indices.json") as f:
    class_indices = json.load(f)
inv_class_indices = {v: k for k, v in class_indices.items()}

def predict_image(path):
    try:
        img = load_img(path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        max_prob = np.max(predictions)
        class_idx = np.argmax(predictions)
        class_name = inv_class_indices[class_idx]

        if max_prob < 0.6:
            return "Image not clear or not recognized as skin", max_prob
        elif class_name == "not_skin":
            return "Invalid image - not a skin image", max_prob
        elif class_name == "no_disease":
            return "No disease found", max_prob
        else:
            return f"Predicted: {class_name}", max_prob
    except Exception as e:
        return f"Error in prediction: {str(e)}", 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None

    if request.method == 'POST':
        if 'image' not in request.files:
            prediction = "No file part in request"
        else:
            file = request.files['image']
            if file.filename == '':
                prediction = "No file selected"
            else:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                prediction, confidence = predict_image(filepath)

    return render_template('index.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)