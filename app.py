from flask import Flask
from flask import render_template, request, url_for
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import FFWebAppFilePaths as filePaths
from keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", imgURL='',modelPrediction="None, not an Image")

@app.route("/imageRecieved")
def recieveImg():
    args=request.args
    imgURLRecieved = args.get('imgUrlInput')
    if not imgURLRecieved:
        return render_template("index.html", imgURL='', modelPrediction="None, not an Image")
    else:
        #need to preprocess incoming image first
        response = requests.get(imgURLRecieved)
        img = Image.open(BytesIO(response.content))

        img = img.resize((224,224)) #resize for vgg16
        img_arry = np.array(img) #turn into 1D array
        img_arry = np.expand_dims(img_arry, axis=0)

        img_processed = preprocess_input(img_arry)


        model = load_model(filePaths.modelPath)
        prediction = model.predict(img_processed)
        predicted_class = np.argmax(prediction)
        class_labels = {0: 'Forest Fire', 1: 'Not Forest Fire'}
        predicted_label = class_labels[predicted_class]

        return render_template("index.html", imgURL=imgURLRecieved, modelPrediction=predicted_label)
    
@app.route("/toModel")
def toModel():
    return render_template("modelStats.html")