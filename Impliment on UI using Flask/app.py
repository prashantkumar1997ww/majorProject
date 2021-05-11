import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, flash, session
from flask_session import Session
import pickle, os, glob
import keras
import librosa
from unittest import result
from librosa.feature.spectral import mfcc
from flask.helpers import url_for


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'wav'}

sess = Session()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

filename = 'mlp_emotion_gender.h5'
model = pickle.load(open(filename, 'rb'))
files = []


@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        audio = []
        if 'audio' not in request.files:
            return 'No file part'
            
        file = request.files['audio']
        if file.filename == '':
            return 'No selected file'

        if file and allowed_file(file.filename):
            filename = "test.wav"    #file.filename

        file.save(os.path.join('audio', filename))
        for x in glob.glob("./audio/*.*"):
            files.append(os.path.basename(x))
            audio.append(extract_feature(x, mfcc=True, chroma=True, mel=True))
        audio = np.array(audio)
        
        result = model.predict(audio)
        ans = result[0].split("_")

        return render_template('index.html', prediction_text='Emotion of person is {} and Gender of person is {}'.format(ans[0], ans[1]))

    else:
        return render_template('index.html')

app.secret_key = 'mehul loves rishabh'

if __name__ == "__main__":
    
    app.run(debug=True)
    app.config['SESSION_TYPE'] = 'filesystem'

    sess.init_app(app)

    app.debug = True
    app.run()