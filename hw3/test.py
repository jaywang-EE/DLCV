import sys
import os
from keras.layers import *#Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from keras.models import Model
from keras.optimizers import SGD
import numpy as np
import random
import scipy
import matplotlib.image as mpimg
#DEBUG
import matplotlib.pyplot as plt
from models import FCN32, VGGUnet
from utils import *


global Ntrain
#Ntrain = 257

Structure = "FCN32"
modelName = "FCN23.hdf5"

#validPath = "datasetTV/validation/"


def main():
    if len(sys.argv) > 1:
        validPath = sys.argv[1]# + '/'
    if len(sys.argv) > 2:
        predictPath = sys.argv[2]# + '/'
    #IMGREAD
    readThrough = True
    
    file_list = [file for file in os.listdir(validPath) if file.endswith('.jpg')]
    file_list.sort()
    n_masks = len(file_list)
    jpg = []
    for i, file in enumerate(file_list):
        '''
        print(i)
        print(file)
        '''
        fileAddr = validPath + file
        readThrough = readThrough and TryImage(fileAddr)
        if readThrough:
            jpg.append(mpimg.imread(fileAddr)[np.newaxis,:])
    '''
    if readThrough:
        print("IMGREAD")
    '''
    x_test = np.concatenate(jpg, axis=0)

    model = FCN32(7)

    model.load_weights(modelName, by_name=True)
    '''
    print("LOADED",modelName)
    '''
    pred = model.predict(x_test, batch_size=8)
    predShape = pred.shape
    #print("predShape", predShape[1:])
    for i, file in enumerate(file_list):
        #print(os.path.splitext(file)[0] + '.jpg')
        array = pred[i]
        image = array.reshape(predShape[1:])
        image = int2dig(image)
        mpimg.imsave(predictPath + (file.split('_')[0]) + '_mask.png', image)
        '''
        if i%100 == 10:
            print(predictPath + os.path.splitext(file)[0] + '.png', image.shape)
        '''
    
     
if __name__ == "__main__":
    main()