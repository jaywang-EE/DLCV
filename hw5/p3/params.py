import torch


VERSION = "S2S_060315"
#MODEL
verbose = 1
#batch_size = 128
ROW = 240
COL = 320
#dimTotal = 64*64*3
TIMESTEP = 400
nClass = 11
#SWITCH
#IsPrint = True
IsTest = True
IsVAE = True
IsValid = False
#TRAIN
EPOCHS = 10
EPOCHSSUB = 10
structures = ['RNN1']
structure = structures[0]

BATCHSIZE = 8
BLOCKSIZE = 1800
val_ratio = 0.1
OPT = 'adadelta'#'rmsprop'
LOSS = 'categorical_crossentropy'#'mean_squared_error'#'binary_crossentropy'

LOSSNAMES = ['loss', 'acc']

ISLOAD = False
MODELDEFAULT = ""

#TRAIN
maxTime = 16

#TEST
modelPath = "models/8S2S_060315.hdf5"
