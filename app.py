# -*- coding: utf-8 -*-

import pickle 
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

with open('./lm.pickle', 'rb') as model_file:
    model = pickle.load(model_file)
    
    
app = Flask(__name__) 

@app.route("/")
@app.route("/lm_flask")
def index():
	return render_template('lm_flask.html')

@app.route('/predict', methods=["GET","POST"])
def predict_mpg():
    if request.method == 'POST':
        cylinders = request.form.get('cylinders')
        weight = request.form.get('weight')
        age = request.form.get('age')
        
        #standardizing the input feature
        sc = StandardScaler()
        data = sc.fit_transform(np.array([[cylinders,weight,age]]))
        prediction = model.predict(data)
    
        return render_template('predictPage.html' , response = str(prediction))
    
 
@app.route('/predict_file', methods=['POST'])
def predict_mpg_from_file():
    if request.method  == 'POST':
        input_data =pd.read_csv(request.files.get('auto_mpg_inputs'), header=None)
    
        #standardizing the input feature
        sc = StandardScaler()
        input_data = sc.fit_transform(np.array(input_data))
        prediction = model.predict(input_data)
        return render_template('predictPage.html' , response = (str(np.array((prediction)))))

if __name__ == '__main__':
    app.run()
    