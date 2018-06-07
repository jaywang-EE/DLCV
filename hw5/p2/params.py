import torch


VERSION = "RNN_060110"
#MODEL
INPUTSHAPE = (24, 512)
verbose = 1
#batch_size = 128
ROW = 240
COL = 320
#dimTotal = 64*64*3
ChannelEnc = 512
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

BATCHSIZE = 16
BLOCKSIZE = 1800
val_ratio = 0.1
OPT = 'adadelta'#'rmsprop'
LOSS = 'categorical_crossentropy'#'mean_squared_error'#'binary_crossentropy'

LOSSNAMES = ['loss', 'acc']

save_path = "../models/"#MODEL

ISLOAD = False
MODELDEFAULT = ""

modelPath = "models/10RNN_060110.hdf5"
modelPathCNN = "models/10CNN_060611.hdf5"
maxTime = 16
