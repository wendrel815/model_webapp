#imports 
import pandas as pd
import pickle as pk
from flask import Flask , request
import os

#load the model

model = pk.load(open('model/model.pkl','rb'))

#Flask instanciation
app = Flask(__name__)

#create endponts
@app.route('/predict',methods=['POST'])

#redirect function


def predict():
  test_json = request.get_json(force=True)
  if isinstance(test_json,dict):
    df_raw = pd.DataFrame(test_json , index=[0])
  else:
    df_raw = pd.DataFrame(test_json , columns = test_json[0].keys())
        
  pred = model.predict(df_raw)
    
  df_raw['predictions'] = pred
    
  return df_raw.to_json(orient='records')
    
#starting the api

if __name__=='__main__':
    port = os.environ.get('PORT',5000)
    app.run(host='0.0.0.0' , port=port)
