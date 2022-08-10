from secrets import choice
from matplotlib.font_manager import json_dump
from requests import session
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from tensorflow import keras
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM , Bidirectional,Dense,GlobalMaxPooling1D,Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from snowballstemmer import TurkishStemmer
import flask
from flask import Flask,request,render_template ,jsonify
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import keras_metrics as km
# from fastapi import FastAPI, Request
# from typing import Union
import random


with open(r"intents.json", encoding='utf-8') as file:
    data = json.load(file)
#getting all the data to lists
tags = []
patterns = []
responses={}

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)


for intent in data['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['patterns']:
    patterns.append(lines)
    tags.append(intent['tag'])
#converting to dataframe
data = pd.DataFrame({"patterns":patterns,
                     "tags":tags})
import string
data['patterns'] = data['patterns'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['patterns'] = data['patterns'].apply(lambda wrd: ''.join(wrd))
#tokenize the data
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=2000)#gtihubdan bak
tokenizer.fit_on_texts(data['patterns'])
train = tokenizer.texts_to_sequences(data['patterns'])

#apply padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(train)

#encoding the outputs
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])#tags




input_shape = x_train.shape[1]
print(input_shape)
#define vocabulary
vocabulary = len(tokenizer.word_index)
print("number of unique words : ",vocabulary)
#output length
output_length = le.classes_.shape[0]
print("output length: ",output_length)


i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,10)(i)
#x = Bidirectional(LSTM(10,return_sequences=True)(x))
x = LSTM(10,return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length,activation="softmax")(x)
model  = Model(i,x)
#compiling the model
model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
#training the model
train = model.fit(x_train,y_train,epochs=500)


def sorgulama(tag,cumle):
  if(tag.find('sspe')!=-1 or tag.find('tüberküloz')!=-1 or tag.find('suçiçeği')!=-1 or tag.find('felci')!=-1 or tag.find('pnömokok')!=-1 or tag.find('kızamık')!=-1 or tag.find('kabakulak')!=-1 or tag.find('kızamıkçık')!=-1 or tag.find('difteri')!=-1 or tag.find('hepatita')!=-1 or tag.find('boğmaca')!=-1 or tag.find('tetanoz')!=-1 ):
    if(cumle.find(' belirti')!=-1 or cumle.find(' semptom')!=-1):
      return responses[tag][1]
    elif(cumle.find(' bulaş')!=-1 or cumle.find(' yayıl')!=-1):
      return responses[tag][2]
    elif(cumle.find(' korun')!=-1 or cumle.find(' sakın')!=-1):
      return responses[tag][3]
    else:
      return responses[tag][0]
  else:
    return random.choice(responses[tag])

def haha(cumle):
    texts_p = []
    prediction_input = cumle
  #removing punctuation and converting to lowercase
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)
    cumle=prediction_input
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    #print(prediction_input)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input],input_shape)
    global sess
    global graph
    with graph.as_default():
      set_session(sess)

      output = model.predict(prediction_input)

    output = output.argmax()

    response_tag = le.inverse_transform([output])[0]
    cevap=sorgulama(response_tag,cumle)
    return cevap

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getbot')
def get_bot_response():
  message = request.args.get('msg')
  veri=[]
  sonuc=haha(message)
  veri.append(str(sonuc))
  cvpp={"cevap": str(sonuc)}
  cvp={"data": [cvpp], "success":True,
  "message":"geldi" }
  resp = flask.Response(json.dumps(cvp,ensure_ascii=False))
  resp.headers["Access-Control-Allow-Origin"] = "*"
  return resp;
  #return json.dumps(cvp,ensure_ascii=False)


if __name__ == "__main__":
        app.run(debug=True ,port=8080,use_reloader=False)
        app.config['JSON_AS_ASCII'] = False 
