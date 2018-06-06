#KERAS
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras import metrics
from keras.layers import Input

import sys
import numpy as np
import random
import scipy
import matplotlib.image as mpimg
#DEBUG
import matplotlib.pyplot as plt
from models import *
from utils import *
from params import *
import os

np.set_printoptions(threshold=np.nan)

def main():
    K.tensorflow_backend._get_available_gpus()
    assert len(sys.argv) > 3, "no enough params!"
    
    if len(sys.argv) > 1:
        imagePath = sys.argv[1]
        print("LOADFROM:", imagePath)
    if len(sys.argv) > 2:
        csvPath = sys.argv[2]
        print("LOADFROM:", csvPath)
    if len(sys.argv) > 3:
        predPath = (sys.argv[3])
        print("LOADTO:", predPath)
    Ntrain = -1
    print('*'*20, "VERSION : "+VERSION, '*'*20)
    
#MODEL
#MODEL_CNN
    modelCNN = Sequential()
    modelCNN.add(Dense(1024, activation='relu', input_shape=(2048,),name="dense1"))
    modelCNN.add(Dense(nClass, activation='softmax',name="dense2"))
    modelCNN.summary()
    
    modelCNN.load_weights(modelPathCNN)

    modelCNN.compile(optimizer=OPT, loss=LOSS, \
            metrics=[metrics.categorical_accuracy])
    modelCNN = Model(inputs=modelCNN.input, outputs=modelCNN.get_layer("dense1").output)
#PREPROCESS
    input_tensor = Input(shape=(224, 224, 3))
    base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
    x = base_model.output
    predictions = GlobalAveragePooling2D()(x)
    modelPre = Model(inputs=base_model.input, outputs=predictions)
    
    modelPre.compile(optimizer=OPT, loss=LOSS, \
            metrics=[metrics.categorical_accuracy])
#MODEL_RNN
    model = LSTMBi((24, 2048))
    model.summary()
    model.load_weights(modelPath)
    model.compile(optimizer=OPT, loss=LOSS, \
            metrics=[metrics.categorical_accuracy])
    modelSub = Model(inputs=model.input, outputs=model.get_layer("BiLSTM").output)

#DATA
#READVEDIO
    labelData = getVideoList(csvPath)
    print(labelData.keys())#DEBUG
    if Ntrain > 0:
        trainSize = Ntrain
    else:
        trainSize = (len(labelData['Video_category']))
    print("CSVFILE:", labelData.keys())
    
    x_testRNN = []
    x_testLen = []
    #labels = []
    for i in range(trainSize):
        readThrough = False
        rawData = readShortVideo(imagePath, labelData['Video_category'][i], labelData['Video_name'][i])
        rawData = centre_crop(rawData)
        #fulledLabels = np.full((rawData.shape[0]), int(labelData['Action_labels'][i]))
        x_testLen.append(rawData.shape[0])
        if i==0:
            x_testCNN = rawData
            #y_target =  fulledLabels
        #if i < 50:
        else:
            x_testCNN = np.concatenate((x_testCNN, rawData),axis=0)
            #y_target  = np.concatenate((y_target , fulledLabels),axis=0)
        #labels.append(int(labelData['Action_labels'][i]))
        
        if i%100 == 0 and i != 0:
            print("LOADING X...", i)
            #print("LOADING Y...", i)
        readThrough = True
    #x_testRNN = keras.preprocessing.sequence.pad_sequences(x_testRNN, maxlen=24)
    dataSize = len(x_testLen)#517

    if readThrough:
        print("dataSize:", dataSize)
        print('*'*20, "VEDIOREAD", '*'*20)
#PREDICT
#CNN
    print("CNN I/P shape: ", x_testCNN.shape)
    #print("CNN labels shape: ", y_target.shape)
#RNN
    #shape = x_testRNN.shape
    features = modelPre.predict(x_testCNN)
    x_testRNN = []
    start = 0
    for i in range(len(x_testLen)):
        end = start + x_testLen[i]
        rawData = features[start:end]
        if rawData.shape[0] > int(maxTime*1.5):
            rawData = halfFilm(rawData)
        x_testRNN.append(rawData)
        start = end
    x_testRNN = keras.preprocessing.sequence.pad_sequences(x_testRNN, maxlen=24)
    print("RNN I/P shape: ", x_testRNN.shape)
    y_testRNN = model.predict(x_testRNN)
    print("RNN O/P shape: ", y_testRNN.shape)
#P1
    f = open(predPath+'p2_result.txt','w')
    filmLabels = np.zeros(dataSize)
    #acc = 0.
    for i in range(dataSize):
        predictLabel = np.argmax(np.mean(y_testRNN[i][-x_testLen[i]:],axis=0))
        f.write(str(predictLabel)+'\n')
        #if labels[i] == int(predictLabel):
        #    acc += 1.
        if i%100 == 0 and i != 0:
            print("LOADING X...", i)

    #acc /= dataSize
    #print(acc)
    f.close()
    
#TSNE
    '''
    print('*'*20, "TSNE", '*'*20)
    
    modelName = "RNN" 
    latent = modelSub.predict(x_testRNN)[:100]
    shape = latent.shape
    latentTmp = latent[0, -x_testLen[0]:]
    print(latent[0,0])
    for i in range(99):
        start = x_testLen[(i+1)]
        latentTmp = np.concatenate((latentTmp, latent[(i+1), -start:]),axis=0)
    latent = latentTmp
    print(latent.shape)
    print('*'*20, "DRAW: "+modelName, '*'*20)
    latent_tsne = tsne(latent, 2)
    plot_scatter(latent_tsne, y_target[:latent.shape[0]], modelName+"NN-based features", predPath+modelName+'.jpg')
    
    modelName = "CNN" 
    latent = modelCNN.predict(features)[:500]
    print('*'*20, "DRAW: "+modelName, '*'*20)
    latent_tsne = tsne(latent, 2)
    plot_scatter(latent_tsne, y_target[:latent.shape[0]], modelName+"NN-based features", predPath+modelName+'.jpg')
    '''
if __name__ == "__main__":
    main()
