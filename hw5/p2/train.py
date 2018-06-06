#KERAS
import keras
from keras.preprocessing import image
from keras import backend as K
from keras import metrics

#from keras.optimizers import SGD
import numpy as np
import random
import scipy
import matplotlib.image as mpimg 
from models import *
from utils import *
from params import *
import sys
import os

#DEBUG
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import gc

np.set_printoptions(threshold=np.nan)

def main():
    K.tensorflow_backend._get_available_gpus()
    np.random.seed(326)
    if len(sys.argv) > 1:
        Ntrain = int(sys.argv[1])
        print("LOAD:", Ntrain)
    else:
        Ntrain = -1
        
    print('*'*20, "VERSION : "+VERSION, '*'*20)
#LABELDEFN
    dictLabel = {}
    file = open(LABLEDEFN, 'r') 
    for line in file:
        tmp = line.split(' ')
        tmp1 = ''
        for i in range(len(tmp)-1):
            tmp1 = tmp1 + tmp[i] + ' '
        dictLabel[tmp1] = int(tmp[len(tmp)-1])
    file.close()
#TRAIN
#READFEATURE
    rawData = np.load(featurePath)
    print(rawData.shape)
    f = open(featureLenPath,'r')  
    filmSize = []
    for line in f.readlines():
        filmSize.append(int(line))  
    f.close()   
#READLABEL
    labelData = getVideoList(labelPath)
    print(labelData.keys())#DEBUG
    trainSize = (len(filmSize))
    if Ntrain > 0 and Ntrain < trainSize:
        trainSize = Ntrain
    print("CSVFILE:", labelData.keys())
#SPLITBYFILM
    start = 0
    end = 0
    x_train = []

    tmp = np.zeros(40)
    for i in range(trainSize):
        tmp[int(filmSize[i])//10]+=1.
    print(tmp)

    for i in range(trainSize):
        readThrough = False
        end = start + filmSize[i]
        arrtmp = rawData[start:end]
        if arrtmp.shape[0] > int(maxTime*1.5):
            arrtmp = halfFilm(arrtmp)
        x_train.append(arrtmp)
        if (i+1)%100 == 0:
            print("LOADING X...", i, x_train[i].shape)
        start = end
        readThrough = True
    x_train = keras.preprocessing.sequence.pad_sequences(x_train)
    y_train = np.zeros(x_train.shape[:2])
    for i in range(trainSize):
        readThrough = False
        y_train[i,:] = int(labelData['Action_labels'][i])
        #np.full(x_train.shape[1], int(labelData['Action_labels'][i]))
        readThrough = True
    y_train = keras.utils.to_categorical(y_train,nClass)
    print("x_train SHAPE:", x_train.shape)
    print("y_train SHAPE:", y_train.shape)
    dataSize = y_train.shape[0]
    if readThrough:
        print('*'*20, "TESTREAD DONE", '*'*20)
#TEST
#READFEATURE
    rawData = np.load(featurePathTest)
    print(rawData.shape)
    f = open(featureLenPathTest,'r')  
    filmSize = []
    for line in f.readlines():
        filmSize.append(int(line))  
    f.close()   
#READLABEL
    labelData = getVideoList(labelPathTest)
    print(labelData.keys())#DEBUG
    trainSize = (len(filmSize))
    if Ntrain > 0 and Ntrain < trainSize:
        trainSize = Ntrain
    print("CSVFILE:", labelData.keys())
#SPLITBYFILM
    start = 0
    end = 0
    x_test = []

    tmp = np.zeros(40)
    for i in range(trainSize):
        tmp[int(filmSize[i])//10]+=1.
    print(tmp)

    for i in range(trainSize):
        readThrough = False
        end = start + filmSize[i]
        arrtmp = rawData[start:end]
        if arrtmp.shape[0] > int(maxTime*1.5):
            arrtmp = halfFilm(arrtmp)
        x_test.append(arrtmp)
        if (i+1)%100 == 0:
            print("LOADING X...", i, x_test[i].shape)
        start = end
        readThrough = True
    x_test = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=x_train.shape[1])
    y_test = np.zeros(x_test.shape[:2])
    for i in range(trainSize):
        readThrough = False
        y_test[i,:] = int(labelData['Action_labels'][i])
        #np.full(x_test.shape[1], int(labelData['Action_labels'][i]))
        readThrough = True
    y_test = keras.utils.to_categorical(y_test,nClass)
    print("x_test SHAPE:", x_test.shape)
    print("y_test SHAPE:", y_test.shape)
    dataSize = y_test.shape[0]
    if readThrough:
        print('*'*20, "TRAINREAD DONE", '*'*20)

#TRAIN
    model = LSTMBi((x_train.shape[1:]))
    model.compile(optimizer=OPT, loss=LOSS, \
            metrics=[metrics.categorical_accuracy])
    
    #model.fit(x_train, y_train, BATCHSIZE, EPOCHS, verbose=1, validation_data=(x_test, y_test))
    
    for ep in range(EPOCHS):
        model.fit_generator(generate_batch_data_random(x_train, y_train, BATCHSIZE),
            steps_per_epoch=len(y_train)//BATCHSIZE*BATCHSIZE,
            nb_epoch=1, 
            verbose=verbose)
#FSTVALID
        score = model.evaluate(x_test,y_test,batch_size=BATCHSIZE)
        print(model.metrics_names)
        print(score)
        #print(np.argmax(np.mean(model.predict(x_test[:20],batch_size=BATCHSIZE),axis=1),axis=1))
        #print(np.argmax(np.mean(y_test[:20],axis=1),axis=1))
        
        label1 = np.argmax(np.mean(model.predict(x_test,batch_size=BATCHSIZE),axis=1),axis=1)
        label2 = np.argmax(np.mean(y_test,axis=1),axis=1)
        print(label1.shape)
        print(label2.shape)
        acc = 0.
        for idx in range(y_test.shape[0]):
            if label1[idx] == label2[idx]:
                acc += 1.
        print(acc/y_test.shape[0])
        
        if ep%3==2:
            print("SAVING:", ep)
            model.save_weights(save_path +str(ep)+VERSION+'.hdf5')
    
    model.save_weights(save_path +str(EPOCHS)+VERSION+'.hdf5')
if __name__ == "__main__":
    main()