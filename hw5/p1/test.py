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
#READVEDIO
    labelData = getVideoList(csvPath)
    print(labelData.keys())#DEBUG
    if Ntrain > 0:
        trainSize = Ntrain
    else:
        trainSize = (len(labelData['Video_category']))
    print("CSVFILE:", labelData.keys())
    
    x_test = []
    #y_test = []
    for i in range(trainSize):
        readThrough = False
        if i%100 == 0 and i != 0:
            print("LOADING X...",i)
        rawData = readShortVideo(imagePath, labelData['Video_category'][i], labelData['Video_name'][i])
        rawData = centre_crop(rawData)

        x_test.append(rawData)
        #y_test.append(int(labelData['Action_labels'][i]))

        if i%100 == 0 and i != 0:
            print("LOADING X...", i, len(x_test))
            #print("LOADING Y...", i, len(y_test))
        readThrough = True

        dataSize = len(x_test)#517

    if readThrough:
        print("dataSize:", dataSize)
        print('*'*20, "VEDIOREAD", '*'*20)
#MODEL
    input_tensor = Input(shape=(224, 224, 3))
    base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    modelPre = Model(inputs=base_model.input, outputs=x)
    
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(2048,)))
    model.add(Dropout(0.5))
    model.add(Dense(nClass, activation='softmax'))
    
    model.compile(optimizer=OPT, loss=LOSS, \
            metrics=[metrics.categorical_accuracy])
#LOAD
    model.load_weights(modelPath)
    model.summary()
#predict
    f = open(predPath+'p1_valid.txt','w')
    filmLabels = np.zeros(dataSize)
    acc = 0.
    for i in range(dataSize):
        features = modelPre.predict(x_test[i], batch_size=8)
        onehotLabels = model.predict(features, batch_size=8)
        '''
        top3 = np.argsort(onehotLabels,axis=1)[:,-3:]
        for j in range(onehotLabels.shape[0]):
            for k in range(3):
                onehotLabels[j][top3[j][k]] = onehotLabels[j][top3[j][k]]*100/(4-k)
        onehotLabels = onehotLabels/100
        '''
        predictLabel = np.argmax(np.mean(onehotLabels,axis=0))
        f.write(str(predictLabel)+'\n')
        #if y_test[i] == int(predictLabel):
        #    acc += 1.
        if i%100 == 0 and i != 0:
            print("LOADING X...", i)

    acc /= dataSize
    print(acc)
    f.close()

if __name__ == "__main__":
    main()