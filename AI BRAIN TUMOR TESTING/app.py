import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model =load_model('CNNMODELTRAINED.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    if classNo == 0:
        return "notu"
    elif classNo == 1:
        return "glioma"
    elif classNo == 2:
        return "pituitary"
    elif classNo == 3:
        return "meningioma"
    
def getResult(img):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (150, 150))  # Resize to the model's expected input shape
    image = image / 255.0  # Normalize to [0, 1]

    input_img = np.expand_dims(image, axis=0)
    result = np.argmax(model.predict(input_img), axis=-1)
    return result



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        print("value nya adalah:  ",value)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)