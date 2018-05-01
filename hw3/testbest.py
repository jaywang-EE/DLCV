import sys
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


IsPrint = True
IsTest = True

global Ntrain
Ntrain = 257

Structure = "FCN32"
modelName = "models/modelO"

validPath = "datasetTV/validation/"

FILEPATH = "/"

def main():
    '''
    if len(sys.argv) > 2:
        print("GET #TRAIN:", sys.argv[2])
        Ntrain = int(sys.argv[2])
    '''
    if len(sys.argv) > 1:
        print("GET MODEL:", sys.argv[1])
        modelName = sys.argv[1]
    #IMGREAD
    readThrough = True
    jpg = []
    for i in range(Ntrain):
        jpgName = validPath + str(i).zfill(4) + "_sat.jpg"
        readThrough = readThrough and TryImage(jpgName)
        if readThrough:
            jpg.append(mpimg.imread(jpgName)[np.newaxis,:])

    if readThrough:
        print("IMGREAD")
    
    x_test = np.concatenate(jpg, axis=0)

    if Structure == "FCN32":
        model = FCN32(Nchannels)
    elif Structure == "Unet":
        model = VGGUnet(Nchannels)
    
    model.load_weights(modelName, by_name=True)
    print("LOADED",modelName)
    
    pred = model.predict(x_test, batch_size=8)
    predShape = pred.shape
    print("predShape", predShape[1:])
    
    predictPath = "datasetTV/predict/"
    for i in range(predShape[0]):
        array = pred[i]
        image = array.reshape(predShape[1:])
        image = int2dig(image)
        mpimg.imsave(predictPath + str(i).zfill(4) + '_mask.png', image)
        if i%100 == 10:
            print(predictPath + str(i).zfill(4) + '_mask.png', image.shape)
        
    
     
if __name__ == "__main__":
    main()