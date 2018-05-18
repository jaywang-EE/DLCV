import torch


VERSION = "VAE_051811"
#MODEL
global Langda
Langda = 1#0.00001
filters = 32
batch_size = 128
ROW = 64
COL = 64
dimTotal = 64*64*3
Iscuda = torch.cuda.is_available()
ChannelEnc = 512
#SWITCH
IsPrint = True
IsTest = False
IsValid = False
#TRAIN
EPOCHS = 30
structures = ['VAE','AE']
structure = structures[0]
printMode = 1

BATCHSIZE = 8
PRINTSIZE = 5
val_ratio = 0.1
OPT = 'adadelta'#'rmsprop'
LOSS = 'mean_squared_error'

save_path = "../models/VAEt/"#MODEL
save_weights_path = "../models/VAEt/"+VERSION#CP

ISLOAD = False
MODELDEFAULT = ""

trainPath = "../data/train/"

#TEST
testPath = "../data/test/"
predictPath = "../pred/VAEt/"