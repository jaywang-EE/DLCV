#Azure
from keras.layers import *#Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from models import FCN32, VGGUnet, FCN8
#from keras.optimizers import SGD
import numpy as np
import random
import scipy
import matplotlib.image as mpimg 
from utils import *
import sys

#DEBUG
import matplotlib.pyplot as plt

K.tensorflow_backend._get_available_gpus()

VERSION = "5_1_2240"

IsPrint = True
IsTest = False
IsValid = False
IsWeighted = True

global Ntrain

EPOCHS = 2
BATCHSIZE = 8
Nchannels = 7
val_ratio = 0.1
OPT = 'adadelta'

FILEPATH = ""
StructureList = ["Unet","FCN32","FCN8"]
Structure = StructureList[1]
save_path = "models"

IsLoad = True
load_model = "models/FCNw22.hdf5"#FCN15.hdf5"

weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"

printMode = 1

def main():
    modelName = '/'
    if len(sys.argv) > 1:
        modelName+=sys.argv[1]
    else:
        modelName+='model'
        
    if len(sys.argv) > 2:
        Ntrain = int(sys.argv[2])
    else:
        Ntrain = 2313
    
    print('*'*30, "VERSION"+VERSION, '*'*30)
    #IMGREAD
    readThrough = True
    jpg = []
    y_train = np.zeros((Ntrain, 512, 512))
    for i in range(Ntrain):
        trainPath = "datasetTV/train/"
        jpgName = FILEPATH + trainPath + str(i).zfill(4) + "_sat.jpg"
        pngName = FILEPATH + trainPath + str(i).zfill(4) + "_mask.png"
        readThrough = readThrough and TryImage(jpgName)
        if readThrough:
            jpg.append(mpimg.imread(jpgName))#[np.newaxis,:])
        readThrough = readThrough and TryImage(pngName)
        if readThrough:
            y_train[i] = (dig2int(mpimg.imread(pngName)))
        
        if i%100 == 0 and i != 0:
            print("LOADING...",i)
        
    if readThrough:
        print('*'*30, "IMGREAD", '*'*30)
    x_train = np.array(jpg)#np.concatenate(jpg, axis=0)
    print(x_train.shape)
    #y_train = np.array(png)
    print(y_train.shape)
    y_train = to_categorical(y_train, Nchannels)
    
    if IsTest:
        y_train = y_train[0:10]
        print(y_train.shape)
        '''
        labels = np.unique(y_train[0:10,:])
        print(labels)
        print(np.unique(y_train.reshape(10*512*512,Nchannels),axis=0))
        print(np.unique(y_train.reshape(10*512*512,Nchannels),axis=0).shape)
        '''
        print(np.sum(y_train.reshape(10*512*512,Nchannels),axis=0))
        weights = np.log10(np.sum(y_train)/np.sum(y_train.reshape(10*512*512,Nchannels),axis=0))
        print(weights)
        weights = weights/min(weights)
        print(weights)
    else:
        #CP
        print('*'*32, "LOADMODEL", '*'*32)
        if Structure == "FCN32":
            model = FCN32(Nchannels)
        elif Structure == "Unet":
            model = VGGUnet(Nchannels)
        elif Structure == "FCN8":
            model = FCN8(Nchannels)
        
        if IsLoad:
            model.load_weights(load_model)
        else:
            model.load_weights(weights_path, by_name=True)
        
        print('*'*32, "TRAIN", '*'*32)
        if IsWeighted:
            print(np.sum(y_train.reshape(Ntrain*512*512,Nchannels),axis=0))
            weights = (np.log10(np.sum(y_train)/np.sum(y_train.reshape(Ntrain*512*512,Nchannels),axis=0)))
            print(weights)
            weights = weights/min(weights)
            '''
            weights = np.log10(weights)
            weights = weights/min(weights)
            weights = weights * weights
            print(weights)
            '''
            model.compile(loss=weighted_categorical_crossentropy(weights),\
                    optimizer= 'adadelta', metrics=['accuracy', 'categorical_accuracy'])
        else:    
            model.compile(loss='categorical_crossentropy',\
                    optimizer= 'adadelta', metrics=['accuracy', 'categorical_accuracy'])
        if IsValid:
            '''
            save_weights_path = "models/checkpoint/"+VERSION
            for ep in range(EPOCHS):
                model.fit_generator(generate_batch_data_random(x_train,\
                        y_train, BATCHSIZEFIT), 512, epochs=1 )
                model.save_weights( save_weights_path + "_" + str( ep ) +'.hdf5' )
            '''
            filepath = "models/checkpoint/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,\
                    save_best_only=True, mode='max')
            callbacks_list = [checkpoint]
            model.fit(x_train, y_train, validation_split=val_ratio, epochs=EPOCHS, batch_size=BATCHSIZE, \
                    callbacks=callbacks_list, verbose=printMode)
        else:
            SaveSap = 4
            save_weights_path = "models/checkpoint/"+VERSION
            for ep in range(EPOCHS):
                model.fit(x_train, y_train, epochs=1,\
                        batch_size=BATCHSIZE, verbose=printMode)
                if True:
                    model.save_weights( save_weights_path + "_" + str( ep ) +'.hdf5' )
                    print("SAVING:", ep)
            
        model.save_weights(save_path+modelName+str(EPOCHS)+VERSION+'.hdf5')
        
    

if __name__ == "__main__":
    main()