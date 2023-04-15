# Dependencies Setup

import pandas as pd
from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import requests
import numpy as np
import pickle as p
import json

# Create an instance of the Flask class
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("main.html")




@app.route("/model")
def getmodels():
    return render_template("model.html")

@app.route('/predict', methods=["GET", "POST"])
def predict():

    if request.method == "POST": 

        # getting user input from HTML form
        hour = request.form["hour"]
        road_class = request.form["road_class"]
        traffic_ctl = request.form["traffic_control"]
        visibility = request.form["visibility"]
        light = request.form["light"]
        condition = request.form["condition"]

        # encoding string input to binary value arrays for the model prediction
        hour_array = onehot_encode("hour", hour)
        road_class_array = onehot_encode("road_class", road_class)
        traffic_ctl_array = onehot_encode("traffic_control", traffic_ctl)
        visibility_array = onehot_encode("visibility", visibility)
        light_array = onehot_encode("light", light)
        condition_array = onehot_encode("condition", condition)

        # joining the arrays into one with 66 features
        data_array = [hour_array + road_class_array + traffic_ctl_array + visibility_array + light_array + condition_array]

 
    modelfile = 'Models/DecisionTree_final_model.pickle'
    model = p.load(open(modelfile, 'rb'))
    predict = model.predict(data_array)

    return render_template("prediction.html", pred = predict[0])

# function to one hot encode user input for model prediction
def onehot_encode(feature, value):

    if feature == "time":
        array = [236, 236, 2040, 236, 236, 1000, 236, 1000, 236, 1000, 236, 1357, 236, 2330, 1020, 1357, 1357, 1730, 1730]
        options = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23"]
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "road_class":
        array = [0,0,0,0,0,0,0,0]
        options = ["Collector", "Expressway", "Laneway", "Local", "Major Arterial", "Major Arterial Ramp", "Minor Arterial", "Pending"]
        for i in range(len(options)):
            if options[i] == value:
                array[i] = 1
    elif feature == "district":
        array = [0,0,0,0,0,0,0,0,0,0,0]
        options =  ["Toronto and East York",
            "Toronto and East York",
            "Etobicoke York",
            "Toronto and East York",
            "Toronto and East York",
            "Scarborough",
            "Toronto and East York",
            "Scarborough",
            "Toronto and East York",
            "Scarborough",
            "Toronto and East York",
            "Toronto and East York",
            "Toronto and East York",
            "Toronto and East York",
            "North York",
            "Toronto and East York",
            "Toronto and East York",
            "Etobicoke York",
            "Etobicoke York",
            "Etobicoke York",
            "Etobicoke York",
            "Etobicoke York",
            "Toronto and East York",
            "Etobicoke York",
            "Etobicoke York",
            "Etobicoke York",
            "Scarborough",
            "Scarborough"]

        for i in range(len(options)):
            if options[i] == value:
                index = i
                array[index] = 1  
    elif feature == "visibility":
        array = [0,0,0,0,0,0,0]
        options = ["Clear", "Drifting Snow", "Fog, Mist, Smoke, Dust", "Freezing Rain", "Rain", "Snow", "Strong wind"]
        for i in range(len(options)):
            if options[i] == value:
                index = i
                array[i] = 1   
    elif feature == "light":
        array = [0,0,0,0,0,0,0,0]
        options = ["Dark", "Dark, artificial", "Dawn", "Dawn, artificial", "Daylight", "Daylight, artificial", "Dusk", "Dusk, artificial"]
        for i in range(len(options)):
            if options[i] == value:
                index = i
                array[i] = 1
    elif feature == "condition":
        array = [0,0,0,0,0,0,0,0]
        options = ["Dry", "Ice", "Loose Sand or Gravel", "Loose Snow", "Packed Snow", "Slush", "Spilled liquid", "Wet"]
        for i in range(len(options)):
            if options[i] == value:
                index = i
                array[i] = 1

    return array

if __name__ == '__main__':
    app.run(debug=True)