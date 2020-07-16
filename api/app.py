import os
import sys
from flask import Flask, request, render_template
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = '/Falling_detection/Models/model_vgg19.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    prediction = model.predict(x)
    prediction = np.argmax(prediction, axis=1)
    if prediction == 0:
        prediction = "Falling"
    else:
        prediction = "No Falling"

    return prediction


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, './Falling_detection/Uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
