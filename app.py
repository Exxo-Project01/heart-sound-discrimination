#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import base64
import numpy as np
import os
import librosa
import tensorflow as tf
from firebase import Firebase
from keras.models import load_model
from pydub import AudioSegment
import soundfile as sf
import io
from six.moves.urllib.request import urlopen


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
storage = firebase.storage()


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
    
    print("Preprocessing the audio...")
    data_mono = preprocess_audio(audio_path)
    print("End the audio.")
    
    print("Start classifying the audio...")
    result = predict_class_of_the_audio_file(data_mono)
    print("Classified the audio.")
    
    #amp_vals = [str(i) for i in amplitude_loader(audio_path)]
    print(result)
    #return result
    return jsonify({
    "user_id": user_id,
    "result": str(result[0])
})


# In[ ]:


def get_audio_from_the_db(user_id):
    users = db.child("users").get()
    
    audio_link = users.val()[user_id]['audio']
    
    # decode base64 string to original binary sound object
    
#     mp3_data = base64.b64decode(b64_str)
#     #print(mp3_data)
#     audio_name = "{}_audio.txt".format(user_id)
#     save_audio = "./sample_audio/{}".format(audio_name)
#     fnew = open(save_audio, "wb")
#     fnew.write(mp3_data)
#     fnew.close()
#     path = convertMp4Towav(save_audio)
    return audio_link


# In[ ]:


# def convertMp4Towav(pathmp4):
#     root = "./sample_audio/"
#     wav_filename = os.path.splitext(os.path.basename(pathmp4))[0] + '.wav'
#     print(pathmp4)
#     wav_path = root+wav_filename
#     AudioSegment.from_file(pathmp4).export(wav_path, format='wav')
#     print(wav_path)
#     os.remove(pathmp4)
#     return wav_path


# In[ ]:


def preprocess_audio(url):
    #url = "https://raw.githubusercontent.com/librosa/librosa/master/tests/data/test1_44100.wav"
    data2, samplerate2 = sf.read(io.BytesIO(urlopen(url).read()),always_2d=False,dtype='float32')
    frame_duration = int(2.97 * samplerate2)
    #print(data2)
    data, samplerate = sf.read(io.BytesIO(urlopen(url).read()),always_2d=False,dtype='float32',frames=frame_duration)
    data_mono = []
    for i in data:
        data_mono.append(i[0])
    
    data_mono_numpy= np.array(data_mono)
    data_mono_numpy = librosa.resample(data_mono_numpy, samplerate, 22050)
    return data_mono_numpy


# In[ ]:


def predict_class_of_the_audio_file(data_mono_numpy):
    #y, sr = librosa.load('test.wav', duration=2.97) 
    #print(y)
    T = [] # Dataset
    ps = librosa.feature.melspectrogram(y=data_mono_numpy, sr=22050)
    print("Shape of the audio file: {}".format(ps.shape))
    if ps.shape != (128, 128): return "Sorry we cannot identify the pattern."
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


#print(predict())
#ipd.Audio(data_mono_numpy, rate=22500)


# In[ ]:


if __name__ == '__main__':
    app.run()


# In[ ]:




