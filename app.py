import numpy as np
import sklearn as sk
import pickle
import json
import tensorflow.keras as keras
import math
import librosa
import streamlit as st
from collections import Counter
loaded_model = pickle.load(open('new_trained_model2.sav','rb'))
#creating a function
def fun(file_path):
    signal, sample_rate = librosa.load(file_path, sr=22050)
    samples_per_segment = int(22050*30)/10
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / 512)
    data = []
    # process all segments of audio file
    for d in range(10):

        # calculate start and finish sample for current segment
        start = int(samples_per_segment * d)
        finish = int(start + samples_per_segment)

        # extract mfcc
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=40, n_fft =2058,
                                    hop_length=512)
        mfcc = mfcc.T

        # store only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data.append(mfcc.tolist())
    x = np.array(data)
    x = x[..., np.newaxis]
    prediction = loaded_model.predict(x)
    #print(prediction)
    predicted_index = np.argmax(prediction, axis=1)
    #print(predicted_index)
    return predicted_index

def main():
    st.title('MUSIC GENERE CLASSIFIER')
    uploaded_file = st.file_uploader("Choose an audio file...", type="wav")
    if uploaded_file is not None:
        # Make predictions using your deep learning model
        prediction = fun(uploaded_file)
        cmp = Counter()
        for i in prediction:
            cmp[i] += 1
        predicted_value = max(cmp, key=cmp.get)
        if(predicted_value==0):
            st.write("The song is of Genre: Blues")
        elif(predicted_value==1):
            st.write("The song is of Genre: Classical")
        elif (predicted_value==2):
            st.write("The song is of Genre: Country")
        elif (predicted_value==3):
            st.write("The song is of Genre: Disco")
        elif (predicted_value==4):
            st.write("The song is of Genre: Hiphop")
        elif (predicted_value ==5):
            st.write("The song is of Genre: Jazz")
        elif (predicted_value==6):
            st.write("The song is of Genre: Metal")
        elif (predicted_value ==7):
            st.write("The song is of Genre: Pop")
        elif (predicted_value ==8):
            st.write("The song is of Genre: Reggae")
        elif (predicted_value==9):
            st.write("The song is of Genre: Rock")
        # Display the results
if __name__== '__main__':
    main()