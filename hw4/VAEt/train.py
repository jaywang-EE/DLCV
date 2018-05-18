#Azure
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

from models import *
#from keras.optimizers import SGD
import numpy as np
import random
import scipy
import matplotlib.image as mpimg 
from utils import *
from params import *
import sys
import os

#DEBUG
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.manual_seed(1)
np.set_printoptions(threshold=np.nan)


def MSE(x, y):
    return torch.mean(torch.pow(x-y, 2))
def KLD(z_mean, z_log_var):

    return -0.5  * torch.mean(1 + z_log_var - torch.pow(z_mean, 2) - torch.exp(z_log_var))

def main():
    np.random.seed(326)
    if len(sys.argv) > 1:
        Ntrain = int(sys.argv[1])
        print("LOAD:", Ntrain)
    else:
        Ntrain = -1

    if len(sys.argv) > 2:
        global Langda
        Langda = float(sys.argv[2])
        print("LOAD:", Langda)
    
    if len(sys.argv) > 3:
        IsLoad = True
        load_model = sys.argv[3]
    else:
        IsLoad = ISLOAD
        load_model = MODELDEFAULT
        
    print('*'*20, "VERSION : "+VERSION, '*'*20)
#IMGREAD
    readThrough = True
    file_list = [file for file in os.listdir(trainPath) if file.endswith('.png')]
    file_list.sort()
    
    if Ntrain > 0:
        trainSize = Ntrain
    else:
        trainSize = len(file_list)
    x_train = np.zeros((trainSize, 64, 64, 3))
    for i, file in enumerate(file_list):
        if Ntrain == i:
            break
        fileAddr = trainPath + file
        readThrough = TryImage(fileAddr) and readThrough
        if readThrough:
            x_train[i] = mpimg.imread(fileAddr)
        else:
            x_train = []
            break
        if i%1000 == 0 and i != 0:
            print("LOADING X...",i)
        
    if readThrough:
        x_train = torch.from_numpy(x_train).double()
        dataloader = Data.DataLoader(x_train, 
                batch_size=BATCHSIZE, shuffle=True, num_workers=2)
    
    if IsTest:
        print(x_train.shape)
    else:
        loss_log = []
        loss_log.append([])
        loss_log.append([])
        loss_log.append([])
#CP
        print('*'*20, "LOADMODEL", '*'*20)
        if structure == "AE":
            encoder = E()
            decoder = D()
            model = AE(encoder, decoder)
            model.cuda()
        elif structure == "VAE":
            encoder = E()
            decoder = D()
            model = VAE(encoder, decoder)
            if Iscuda:
                model.cuda()
        #print(model)

        if IsLoad:
            model.load_state_dict(torch.load(load_model))
            print("LOADMODEL : ", load_model)
        #else:
        #    model.load_weights(weights_path, by_name=True)
#TRAIN
        BinCE = nn.BCELoss()#nn.BCELoss(size_average=False)#nn.MSELoss()
        print('*'*20, "TRAIN", '*'*20)
        numSteps = int(float(trainSize)/BATCHSIZE)
        process_bar = ShowProcess(numSteps)
        optimizer = torch.optim.Adadelta(model.parameters())#, lr=0.00001)#TODO
        for ep in range(EPOCHS):
            l = 0.0
            lk = 0.0
            lr = 0.0
            for i, data in enumerate(dataloader, 0):
                x = Variable(data)
                if Iscuda:
                    x = x.cuda()
                optimizer.zero_grad()
                y = model(x)
                #vlu, y = y.max(4)
                l_Rec = MSE(y, x)
                if Iscuda:
                    l_Rec = l_Rec.cuda()    
                if structure == "VAE":
                    l_KL = KLD(model.z_mean, model.z_log_var)
                    if Iscuda:
                        l_KL = l_KL.cuda()
                else:
                    print("AE")
                    l_KL =  MSE(x, y)
                    if Iscuda:
                        l_KL = l_KL.cuda() 
                loss = l_Rec + l_KL * Langda
                loss.backward()
                optimizer.step()
                l += loss.data[0]
                lr += l_Rec.data[0]
                lk += l_KL.data[0]
#PRINT
                if i%PRINTSIZE==PRINTSIZE - 1:
                    loss_log[0].append(l/PRINTSIZE)
                    loss_log[1].append(lk/PRINTSIZE)
                    loss_log[2].append(lr/PRINTSIZE)
                    process_bar.show_process(l/PRINTSIZE, lk/PRINTSIZE, lr/PRINTSIZE, ep, i)
                    l, lk, lr = (0,0,0)
            start_i = int(ep*numSteps/PRINTSIZE)
            end_i = int((ep + 1)*numSteps/PRINTSIZE)
            l = np.mean(np.array(loss_log[0][start_i:end_i]).astype(float))
            lk = np.mean(np.array(loss_log[1][start_i:end_i]).astype(float))
            lr = np.mean(np.array(loss_log[2][start_i:end_i]).astype(float))
            print(ep+1, l, lk, lr)
        
            x_step = range(0, len(loss_log[1])*PRINTSIZE*BATCHSIZE, PRINTSIZE*BATCHSIZE)
            plt.plot(x_step, loss_log[1])
            plt.xlabel('steps')
            plt.ylabel('loss')
            plt.savefig(str(VERSION)+str(Langda)+'KL_loss.png')
            plt.close('all')
            plt.plot(x_step, loss_log[2])
            plt.xlabel('steps')
            plt.ylabel('loss')
            plt.savefig(str(VERSION)+str(Langda)+'Rec_loss.png')
            plt.close('all')
            
            x = x_train[0:8]
            if Iscuda:
                x = x.cuda()
            y_gpu = model(x)
            y = y_gpu.cpu()
            pred = y.data.numpy()
            for i in range(BATCHSIZE):
                image = np.array(pred[i]*256.).astype(int)
                mpimg.imsave("/home/jay/hw4/pred/VAEt/EP"+str(ep) +'_'+ str(i).zfill(4)+'.png', image)
    
            torch.save(model.state_dict(), save_path+str(VERSION)+'_'+str(Langda)+'params.pkl')
        
        torch.save(model.state_dict(), save_path+str(VERSION)+'_'+str(Langda)+'params.pkl')
        
if __name__ == "__main__":
    main()