import glob
import os
import librosa
import time
import numpy as np
import pandas as pd
from unittest import result


emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fear',
  '07':'disgust',
  '08':'surprised'
}


tess_emotions={
  'neutral':'neutral',
  'happy':'happy',
  'sad':'sad',
  'angry':'angry',
  'fear':'fearful',
  'disgust':'disgust',
  'ps':'surprised'
}


#defined tess emotions to test on TESS dataset only
tess_emotions=['angry','disgust','fear','ps','happy','sad']

##defined RAVDESS emotions to test on RAVDESS dataset only
ravdess_emotions=['neutral','calm','angry', 'happy','disgust','sad','fear','surprised']

observed_emotions = ['sad','angry','happy','disgust','surprised','neutral','calm','fear']

observed_gender = ['male','female']

def extract_feature(file_name, mfcc):
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
    result=""
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    return result

def dataset_options():
    # choose datasets
    ravdess = True
    tess = True
    ravdess_speech = False
    ravdess_song = False
    data = {'ravdess':ravdess, 'ravdess_speech':ravdess_speech, 'ravdess_song':ravdess_song, 'tess':tess}
    return data

def gender_ravdess(g):
    """Returns Gender Label"""
    if int(g[0:2]) % 2 == 0:
        return 'female'
    else:
        return 'male'

def gender_tess(g):
    """Returns Gender Label"""
    if g == 'Y':
        return 'female'
    else:
        return 'male'

def load_data(test_size=0.3): 
    x,y=[],[]
    
    # feature to extract
    mfcc = True
    
    data = dataset_options()
    paths_ravdess = []
    paths_ravdess.append("./Datasets/RAVDESS/*/*/Actor_*/*.wav")
    
    for path in paths_ravdess:
        for file in glob.glob(path):
            file_name=os.path.basename(file)
            print(file_name)
            emotion=emotions[file_name.split("-")[2]] + '_' + gender_ravdess(file_name.split("-")[-1])
            feature=extract_feature(file, mfcc)
            x.append(feature)
            y.append(emotion)
    
    
    paths_tess = []
    paths_tess.append("./Datasets/TESS/*/*.wav")
    
    for path in paths_tess:
        for file in glob.glob(path):
            file_name=os.path.basename(file)
            print(file_name)
            emotion=tess_emotions[int(file_name.split('_')[-1].split('.')[0])] + '_' + gender_tess(file_name[0])
            feature=extract_feature(file, mfcc)
            x.append(feature)
            y.append(emotion)
    

    return {"X":x,"y":y}

start_time = time.time()

Trial_dict = load_data(test_size = 0.3)

print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))


X = pd.DataFrame(Trial_dict["X"])
y = pd.DataFrame(Trial_dict["y"])

print (X.shape, y.shape)

y=y.rename(columns= {0: 'emotion_gender'})

data = pd.concat([X, y], axis =1)

data.head()

data = data.reindex(np.random.permutation(data.index))

data.to_csv("emotion_gender_data.csv")

