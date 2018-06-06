#KERAS
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
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
import cv2

np.set_printoptions(threshold=np.nan)

def main():
    K.tensorflow_backend._get_available_gpus()
    np.random.seed(326)
    
    if len(sys.argv) > 1:
        imagePath = sys.argv[1]
        print("LOADFROM:", imagePath)
    if len(sys.argv) > 2:
        predPath = (sys.argv[2])
        print("LOADTO:", predPath)
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
#PREPROCESS
    input_tensor = Input(shape=(224, 224, 3))
    base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
    x = base_model.output
    predictions = GlobalAveragePooling2D()(x)
    modelPre = Model(inputs=base_model.input, outputs=predictions)
    modelPre.compile(optimizer=OPT, loss=LOSS, \
            metrics=[metrics.categorical_accuracy])
#MODEL
    model = LSTMBi((TIMESTEP, 2048))
    model.compile(optimizer=OPT, loss=LOSS, \
            metrics=[metrics.categorical_accuracy])
    model.load_weights(modelPath)
#READIMAGES&PREDICT
    folders = os.listdir(imagePath)
    for subfolder in folders:
        fullpath = imagePath + subfolder
        assert os.path.isdir(fullpath), "EXISTDOCS?"
        x_train = images2npy(fullpath, Ntrain, True)
#predict
        features = modelPre.predict(x_train, batch_size=8)
        #SPLIT TIMESTEP
        tmpLen = x_train.shape[0]//TIMESTEP
        start = 0
        splitFeatures = []
        for i in range(tmpLen):
            end = start + TIMESTEP
            splitFeatures.append(features[start:end])
            start = end
        if start != x_train.shape[0]:
            tmpArr = np.zeros((TIMESTEP, 2048))
            print(start,x_train.shape[0])
            tmpArr[:x_train.shape[0]-start] = features[start:x_train.shape[0]]
            splitFeatures.append(tmpArr)
        splitFeatures = np.array(splitFeatures)
        labels = model.predict(splitFeatures, batch_size=8)
        labels = np.argmax(np.reshape(labels,((tmpLen + 1)*TIMESTEP,nClass))[:x_train.shape[0]],axis=1)
        #print("PREDICT DONE:",subfolder,features.shape, labels.shape)
        
        f = open(predPath+subfolder+'.txt','w')
        for i in range(labels.shape[0]):
            f.write(str(labels[i])+'\n')
        f.close()
        """
        if x_train.shape[0]==2140:#BEST
            labelAddr = labelPathTest + subfolder + '.txt'
            filmLen = x_train.shape[0]
            
            f = open(labelAddr,'r')  
            target = np.zeros((filmLen))
            count = 0
            for line in f.readlines():
                target[count] = (int(line))
                count += 1
            f.close()
            
            '''
            assert labels.shape[0] == target.shape[0]
            bestIdx = 0
            bestAcc = 0.
            for idx in range(500,filmLen-320-200):
                tmpAcc = calSubACC(labels[idx:idx+320], target[idx:idx+320])
                if tmpAcc > bestAcc:
                    bestIdx = idx
                    bestAcc = tmpAcc
            print(bestAcc, bestIdx)
            ''' 
            file_list = [file for file in os.listdir(fullpath) if file.endswith('.jpg')]
            file_list.sort()
            #print(len(file_list))
            #for j in range(5):
            timeStart = 1300#1072
            timeLen = 320
            timeEnd = timeStart + timeLen
            file_listSub = file_list[timeStart:timeEnd:32]
            predImg = np.zeros((280,3200,3))
            
            outputLabels = labels[timeStart:timeEnd]#/nClass*255
            outputTargets = target[timeStart:timeEnd]#/nClass*255
            #print(calSubACC(outputLabels, outputTargets))
            for i in range(timeLen):
                tmpTar = int(outputTargets[i])
                tmpTar = label2rgb(tmpTar)
                tmpLab = int(outputLabels[i])
                tmpLab = label2rgb(tmpLab)
                for color in range(3):
                    predImg[ :20,i*10:(i+1)*10,color] = np.full((20,10), tmpTar[color])
                    predImg[260:,i*10:(i+1)*10,color] = np.full((20,10), tmpLab[color])
            for i, file in enumerate(file_listSub):
                #print(file)
                fileAddr = os.path.join(fullpath,file)
                try:
                    img = cv2.imread(fileAddr)
                    predImg[20:260,i*320:(i+1)*320] = img
                except:
                    assert False,("ERROR "+fileAddr)
            cv2.imwrite("predShow"+".png", predImg)
        """
if __name__ == "__main__":
    main()