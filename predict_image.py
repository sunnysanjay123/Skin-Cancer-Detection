import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

model = load_model("model.h5")
with open("class_indices.json") as f:
    class_indices = json.load(f)

inv_class_indices = {v: k for k, v in class_indices.items()}

def predict_image(path):
    try:
        img = load_img(path, target_size=(224, 224))  
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        max_prob = np.max(predictions)
        class_idx = np.argmax(predictions)
        class_name = inv_class_indices[class_idx]

        if max_prob < 0.6:
            return "Unknown or unclear image", max_prob
        elif class_name == "not_skin":
            return "Invalid image - not skin", max_prob
        elif class_name == "no_disease":
            return "No disease found", max_prob
        else:
            return class_name, max_prob
    except Exception as e:
        return f"Error: {str(e)}", 0.0