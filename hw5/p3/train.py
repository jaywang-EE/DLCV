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
#READFILE
from os import listdir
from os.path import isfile, isdir, join

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
#TRAINDATA
#READFEATURE
    x_trainRaw = []
    y_trainRaw = [] 
    file_list = [file for file in os.listdir(featurePath) if file.endswith('.npy')]
    #file_list.sort()
    for i, file in enumerate(file_list):
        fileAddr = featurePath+file
        labelAddr = splitext(file)[0]
        labelAddr = labelPath + labelAddr + '.txt'

        rawData = np.load(fileAddr)
        filmLen = rawData.shape[0]
        #print(rawData.shape)

        f = open(labelAddr,'r')  
        labels = np.zeros((filmLen))
        count = 0
        for line in f.readlines():
            if count == filmLen and Ntrain > 0:
                break
            labels[count] = (int(line))
            count += 1
        f.close()
        x_trainRaw.append(rawData)
        y_trainRaw.append(labels)
        print(labels[TIMESTEP:TIMESTEP+40])
        print("LOADING:",fileAddr,labelAddr)

    x_train = []
    y_train = []
    for i in range(len(x_trainRaw)):
        #while end < x_trainRaw[i].shape[0]:
        x_train.append(x_trainRaw[i][0:TIMESTEP])
        y_train.append(y_trainRaw[i][0:TIMESTEP])
        filmLen = x_trainRaw[i].shape[0]
        for _ in range((filmLen*2)//TIMESTEP):
            start = randint(0,filmLen - 1 - TIMESTEP)
            end = start + TIMESTEP
            x_train.append(x_trainRaw[i][start:end])
            y_train.append(y_trainRaw[i][start:end])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = keras.utils.to_categorical(y_train,nClass)
    print(x_train.shape)
    print(y_train.shape)
#TESTDATA
#READFEATURE
    x_trainRaw = []
    y_trainRaw = [] 
    file_list = [file for file in os.listdir(featurePathTest) if file.endswith('.npy')]
    #file_list.sort()
    for i, file in enumerate(file_list):
        fileAddr = featurePathTest+file
        labelAddr = splitext(file)[0]
        labelAddr = labelPathTest + labelAddr + '.txt'

        rawData = np.load(fileAddr)
        filmLen = rawData.shape[0]
        #print(rawData.shape)

        f = open(labelAddr,'r')  
        labels = np.zeros((filmLen))
        count = 0
        for line in f.readlines():
            if count == filmLen and Ntrain > 0:
                break
            labels[count] = (int(line))
            count += 1
        f.close() 
        x_trainRaw.append(rawData)
        y_trainRaw.append(labels)
        print(labels[TIMESTEP:TIMESTEP+40])
        print("LOADING:",fileAddr,labelAddr)

    x_test = []
    y_test = []
    '''
    for i in range(len(x_trainRaw)):
        #while end < x_trainRaw[i].shape[0]:
        x_test.append(x_trainRaw[i][0:TIMESTEP])
        y_test.append(y_trainRaw[i][0:TIMESTEP])
        filmLen = x_trainRaw[i].shape[0]
        for _ in range((filmLen*2)//TIMESTEP):
            start = randint(0,filmLen - 1 - TIMESTEP)
            end = start + TIMESTEP
            x_test.append(x_trainRaw[i][start:end])
            y_test.append(y_trainRaw[i][start:end])
            #start = end
            #end = start + TIMESTEP
    '''
    lenTest = []
    sizeTest = []#np.zeros(len(x_trainRaw))
    for i in range(len(x_trainRaw)):
        tmpSize = 0
        lenTest.append(x_trainRaw[i].shape[0])
        start = 0#randint(0,TIMESTEP - 1)
        end = start + TIMESTEP
        while end < x_trainRaw[i].shape[0]:
            tmpSize += 1
            x_test.append(x_trainRaw[i][start:end])
            y_test.append(y_trainRaw[i][start:end])
            #print(x_test[len(x_test)-1].shape)
            start = end
            end = start + TIMESTEP
        x_test.append(x_trainRaw[i][-TIMESTEP:])
        y_test.append(y_trainRaw[i][-TIMESTEP:])
        sizeTest.append(tmpSize+1)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_test = keras.utils.to_categorical(y_test,nClass)
    print(x_test.shape)
    print(y_test.shape)
    print(lenTest)
#TRAIN
    model = LSTMBi((x_train.shape[1:]))
    model.compile(optimizer=OPT, loss=LOSS, \
            metrics=[metrics.categorical_accuracy])
    
    
    pred = model.predict(x_test,batch_size=BATCHSIZE)
    filmPred = []
    labels = []
    start = 0
    for i in range(len(sizeTest)):
        end = start + int(sizeTest[i])
        filmPred.append(np.reshape(pred[start:end],(sizeTest[i]*TIMESTEP,nClass))[:lenTest[i]])
        labels.append(np.reshape(y_test[start:end],(sizeTest[i]*TIMESTEP,nClass))[:lenTest[i]])
        start = end
    filmAcc = []
    for i in range(len(sizeTest)):
        print(filmPred[i].shape)
        print(labels[i].shape)
        tmp = np.array(np.argmax(filmPred[i],axis=1) - np.argmax(labels[i],axis=1))
        tmpAcc = 0.
        for j in range(tmp.shape[0]):
            if tmp[j] == 0:
                tmpAcc += 1.
        tmpAcc /= tmp.shape[0]
        filmAcc.append(tmpAcc)
    print("ACC:",filmAcc) 
        
        
#TRAIN
    for ep in range(EPOCHS):
        model.fit_generator(generate_batch_data_random(x_train, y_train, BATCHSIZE),
            steps_per_epoch=len(y_train)//BATCHSIZE*BATCHSIZE,#dataSize//BATCHSIZE,
            nb_epoch=1, 
            verbose=verbose)
#FSTVALID
        pred = model.predict(x_test,batch_size=BATCHSIZE)
        filmPred = []
        labels = []
        start = 0
        for i in range(len(sizeTest)):
            end = start + int(sizeTest[i])
            filmPred.append(np.reshape(pred[start:end],(sizeTest[i]*TIMESTEP,nClass))[:lenTest[i]])
            labels.append(np.reshape(y_test[start:end],(sizeTest[i]*TIMESTEP,nClass))[:lenTest[i]])
            start = end
        filmAcc = []
        for i in range(len(sizeTest)):
            print(filmPred[i].shape)
            print(labels[i].shape)
            tmp = np.array(np.argmax(filmPred[i],axis=1) - np.argmax(labels[i],axis=1))
            tmpAcc = 0.
            for j in range(tmp.shape[0]):
                if tmp[j] == 0:
                    tmpAcc += 1.
            tmpAcc /= tmp.shape[0]
            filmAcc.append(tmpAcc)
        print("ACC:",filmAcc) 
        '''
        pred = model.predict(x_test,batch_size=BATCHSIZE)
        filmPred = []
        labels = []
        start = 0
        for i in range(len(sizeTest)):
            end = start + int(sizeTest[i])
            filmPred.append(np.reshape(pred[start:end],(sizeTest[i]*TIMESTEP,nClass))[:lenTest[i]])
            labels.append(np.reshape(y_test[start:end],(sizeTest[i]*TIMESTEP,nClass))[:lenTest[i]])
            start = end
        filmAcc = []
        for i in range(len(sizeTest)):
            print(filmPred[i].shape)
            print(labels[i].shape)
            tmp = np.array(np.argmax(filmPred[i],axis=1) - np.argmax(labels[i],axis=1))
            tmpAcc = 0.
            for j in range(tmp.shape[0]):
                if tmp[j] == 0:
                    tmpAcc += 1.
            tmpAcc /= tmp.shape[0]
        print("ACC:",filmAcc)    
        '''
        print(np.argmax((pred[1:2]),axis=2))
        print(np.argmax((pred[1:2]),axis=2)-np.argmax((y_test[1:2]),axis=2))
        if ep%3==2:
            print("SAVING:", ep)
            model.save_weights(save_path +str(ep)+VERSION+'.hdf5')
        model.save_weights(save_path +str(EPOCHS)+VERSION+'.hdf5')

if __name__ == "__main__":
    main()