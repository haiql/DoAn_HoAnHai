from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image 

# Keras
from keras.optimizers import Adam
from keras.applications.imagenet_utils import  preprocess_input
from tensorflow.keras.models import Model , load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


model = load_model('D:\My Project\DoAn\skin_cancer_model_03.h5')



lesion_classes_dict = {
    0 : 'Melanocytic nevi',
    1 : 'Melanoma',
    2 : 'Benign keratosis-like lesions ',
    3 : 'Basal cell carcinoma',
    4 : 'Actinic keratoses',
    5 : 'Vascular lesions',
    6 : 'Dermatofibroma'
}


def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(75,100, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y = model.predict(x)
    
    return y


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path , model)

        # Process your result for human
        

        pred_class = preds.argmax(axis=-1)            
        #pred_class = decode_predictions(preds, top=3)   
        #result = lesion_classes_dict.get(pred_class[0], 'Unknown')
        pr = lesion_classes_dict[pred_class[0]]
        result =str(pr)         
        return result       
        
    return None


if __name__ == '__main__':
    app.run(debug=True)