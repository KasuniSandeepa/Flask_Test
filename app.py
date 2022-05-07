from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd
from flask_restful import reqparse
import traceback

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return "hey"

@app.route('/predict', methods=['POST'])
def predict():
 lr = joblib.load("model.pkl")
 if lr:
  try:
   json = request.get_json()
   model_columns = joblib.load("model.pkl")
   temp=list(json[0].values())
   vals=np.array(temp)
   prediction = lr.predict(temp)
   print("here:",prediction)
   return jsonify({'prediction': str(prediction[0])})

  except:
   return jsonify({'trace': traceback.format_exc()})
 else:
  return ('No model here to use')


if __name__ == '__main__':
    # app.run()
    app.run(debug=True)
