import torch


VERSION = "GAN_051816"
#MODEL
global Langda
Langda = 1#0.00001
filters = 32
batch_size = 128
ROW = 64
COL = 64
dimTotal = 64*64*3
Iscuda = torch.cuda.is_available()
ChannelEnc = 50
#SWITCH
IsPrint = True
IsTest = False
IsVAE = False
IsValid = False
#TRAIN
EPOCHS = 10
structures = ['GAN']
structure = structures[0]
printMode = 1
global DBias
DBias = 1
global GBias
GBias = 1

BATCHSIZE = 8
PRINTSIZE = 5
val_ratio = 0.1
OPT = 'adadelta'#'rmsprop'

save_path = "/home/jay/hw4/models/GAN/"#MODEL
save_weights_path = "/home/jay/hw4/models/GAN/"+VERSION#CP

ISLOAD = False
MODELDEFAULT = ""

trainPath = "/home/jay/hw4/data/train/"

#TEST
testPath = "/home/jay/hw4/data/test/"
predictPath = "/home/jay/hw4/pred/VAEt/"