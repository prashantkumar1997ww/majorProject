import keras
import numpy as np
import librosa
import pickle, os, glob

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

audio = []
files = []

for file in glob.glob("./examples/*.*"):
    files.append(os.path.basename(file))
    audio.append(extract_feature(file, mfcc=True, chroma=True, mel=True))

audio = np.array(audio)

filename = 'mlp_emotion_gender.h5'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(audio)
for i in range(len(result)):
    print(files[i]+" = "+result[i])
