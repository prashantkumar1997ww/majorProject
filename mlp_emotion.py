import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import re

def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
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


# Emotions in the RAVDESS, TESS & SAVEE dataset
rvdess_emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
# savee_emotions={
#   'n':'neutral',
#   'c':'calm',
#   'h':'happy',
#   's':'sad',
#   'a':'angry',
#   'f':'fearful',
#   'd':'disgust',
#   'p':'surprised'
# }
tess_emotions={
  'neutral':'neutral',
  'happy':'happy',
  'sad':'sad',
  'angry':'angry',
  'fear':'fearful',
  'disgust':'disgust',
  'ps':'surprised'
}
# Emotions to observe
observed_emotions=['neutral','calm','happy','sad','angry','fearful', 'disgust','surprised']

def load_data(test_size):
    x,y=[],[]

    for file in glob.glob("./Dataset/ravdess/*.wav"):
        file_name=os.path.basename(file)
        emotion=rvdess_emotions[file_name.split("-")[2]]

        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    
    # for file in glob.glob("./Dataset/savee/*/*.wav"):
    #     file_name=os.path.basename(file)
    #     f=file_name[0]
    #     # print(f)
    #     emotion=savee_emotions[f]

    #     if emotion not in observed_emotions:
    #         continue
    #     feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
    #     x.append(feature)
    #     y.append(emotion)

    for file in glob.glob("./Dataset/tess/*.wav"):
        file_name=os.path.basename(file)
        f=file_name.split('_')[-1].split('.')[0]
        emotion=tess_emotions[f]

        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, train_size= 0.95,random_state=9)


import time
x_train,x_test,y_train,y_test=load_data(test_size=0.05)

#Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]), x_test.shape[1])

# Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

# Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# Train the model
model.fit(x_train,y_train)

# Predict for the test set
x_pred=model.predict(x_train)
y_pred=model.predict(x_test)
# print(x_pred, y_pred)

# Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,y_pred)
print (matrix)


