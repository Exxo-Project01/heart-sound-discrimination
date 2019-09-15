#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
import base64
import numpy as np
import librosa
import tensorflow as tf
from firebase import Firebase
from keras.models import load_model


# In[ ]:


app = Flask(__name__)


# In[ ]:


# load model
global model
model = load_model('./model/hsv_cnn.hdf5')


# In[ ]:


global graph
graph = tf.get_default_graph()


# In[ ]:


config = {
  "apiKey": "AIzaSyC_uv5-QB8LDGV82HHoMlzKKDlJvA2tDMk ",
  "authDomain": "dengue-20fc0.firebaseapp.com",
  "databaseURL": "https://dengue-20fc0.firebaseio.com/",
  "storageBucket": "dengue-20fc0.appspot.com"
}

firebase = Firebase(config)


# In[ ]:


db = firebase.database()


# In[ ]:


@app.route('/', methods = ['GET', 'POST', 'PATCH', 'PUT', 'DELETE'])
def index():
    return "Welcome to hsv project"


# In[ ]:


@app.route('/predict', methods = ['GET', 'POST', 'PATCH', 'PUT', 'DELETE'])
def predict():
    #user_id = "Joe1234"
    user_id = request.args.get('user_id')
    print("Heart sound disriminator started.")
    print("Getting heart sound file of the user {}...".format(user_id))
    audio_path = get_audio_from_the_db(user_id)
    print("Audio file captured.")
    result = predict_class_of_the_audio_file(audio_path)
    #amp_vals = [str(i) for i in amplitude_loader(audio_path)]
    print(result)
    #return all_result
    return jsonify({
    "user_id": user_id,
    "result": str(result[0]),
    "freq": ""
})


# In[ ]:


def get_audio_from_the_db(user_id):
    users = db.child("users").get()
    #print(users.val()[user_id])
    
    # pick sound file you have in working directory
    # or give full path
    #sound_file = "./heart_sound_discrimination_model_classifire/data/UrbanSound8K/audio/test/207214-2-0-126.wav"
    
    # use mode = "rb" to read binary file
    #fin = open(sound_file, "rb")
    #binary_data = fin.read()
    #fin.close()
    
    # encode binary to base64 string (printable)
    #b64_data = base64.b64encode(binary_data)
    #b64_fname = "original_b64.txt"
    
    # save base64 string to given text file
    #fout = open(b64_fname, "wb")
    #fout.write(b64_data)
    #fout.close
    
    # read base64 string back in
    #fin = open(b64_fname, "r")
    #b64_str = fin.read()
    #fin.close()
    
    b64_str = users.val()[user_id]['audio']
    
    # decode base64 string to original binary sound object
    
    mp3_data = base64.b64decode(b64_str)
    #print(mp3_data)
    save_audio = "./sample_audio/{}_audio.wav".format(user_id)
    fnew = open(save_audio, "wb")
    fnew.write(mp3_data)
    fnew.close()
    return save_audio


# In[ ]:


def predict_class_of_the_audio_file(audio_path):
    T = [] # Dataset
    
    y, sr = librosa.load(audio_path, duration=2.97)  
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    print("Shape of the audio file: {}".format(ps.shape))
    if ps.shape != (128, 128): return
    T.append( ps )
    
    #Reshaping
    test_reshaped = T
    test_reshaped = np.array([x.reshape( (128, 128, 1) ) for x in test_reshaped])
    print("Reshaped audio file: {}".format(test_reshaped.shape))
    
    #Prediction
    with graph.as_default():
        predict = model.predict(x=test_reshaped)
    
        
    #Classes
    #classes = {4:'heart_sound', 2:'children_playing', 3:'dog_bark', 0:'air_conditioner', 1: 'car_horn'}
    class_of_audio = predict.argmax(axis=-1)
    
    return class_of_audio


# In[ ]:


#print(predict_heart_sound())


# In[ ]:


if __name__ == '__main__':
    app.run()


# In[ ]:




