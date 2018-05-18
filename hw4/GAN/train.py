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
'''
def MSE(x, y):
    return torch.mean(torch.pow(x-y, 2))


def KLD(z_mean, z_log_var, ld):

    return -0.5  * torch.mean(1 + z_log_var -\
            z_mean*z_mean - torch.exp(z_log_var))
'''
def main():
    torch.manual_seed(1)
    np.random.seed(326)
    if len(sys.argv) > 1:
        Ntrain = int(sys.argv[1])
        print("LOAD:", Ntrain)
    else:
        Ntrain = -1

    if len(sys.argv) > 2:
        global DBias
        global GBias
        bias = float(sys.argv[2])
        if bias < 1.:#DBias
            bias = 1.0/bias
            DBias = int(bias)
        else:#GBias
            GBias = int(bias)
    print("LOAD DBias:", DBias)
    print("LOAD GBias:", GBias)
    
    if len(sys.argv) > 4:
        IsLoad = True
        load_modelD = sys.argv[3]
        load_modelG = sys.argv[4]
    else:
        IsLoad = ISLOAD
        load_modelG = MODELDEFAULT
        load_modelD = MODELDEFAULT
        
    if len(sys.argv) > 5:
        learningRate = float(sys.argv[2])
    else:
        learningRate = 0.0001
        
    print('*'*20, "VERSION : "+VERSION, '*'*20)
#CP
    if structure == "GAN":
        #decoder = D()
        generator = D()
        discreminator = Discreminator(E())
        if Iscuda:
            generator.cuda()
            discreminator.cuda()
    else:
        print("ERROR: NO MODULE NAMED", structure)
    print(generator)
    print(discreminator)

    if IsLoad:
        discreminator.load_state_dict(torch.load(load_modelD))
        generator.load_state_dict(torch.load(load_modelG))
        print("LOADMODEL : ", load_modelD)
        print("LOADMODEL : ", load_modelG)
    elif IsVAE: #LOADFROM VAE
        vae_model = "../models/VAEt/VAE_051216_1params.pkl"
        encoder = E()
        decoder = D()
        vae = VAE(encoder, decoder)
        if Iscuda:
            vae = vae.cuda()
        tmp = torch.load(vae_model, map_location=lambda storage, loc: storage)
        vae.load_state_dict(tmp)#, map_location=lambda storage, loc: storage)
        #pretrained_dict = vae.state_dict()
        generator = vae.decoder
        discreminator = Discreminator(vae.encoder)
        if Iscuda:
            discreminator = discreminator.cuda()
    print('*'*20, "LOADMODEL", '*'*20)
    print(generator)
    print(discreminator)
        
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
        print(np.max(x_train))
        x_train = torch.from_numpy(x_train).double()
        dataloader = Data.DataLoader(x_train, 
                batch_size=BATCHSIZE, shuffle=True, num_workers=2)

    loss_log = []
    loss_log.append([])
    loss_log.append([])
    loss_log.append([])
    loss_log.append([])
#TRAIN
    print('*'*20, "TRAIN", '*'*20)
    numSteps = int(float(trainSize)/BATCHSIZE)
    process_bar = ShowProcess(numSteps)
    
    binCE = nn.BCELoss()
    '''
    optimizerG = torch.optim.RMSprop(generator.parameters(), lr=0.0002)
    optimizerD = torch.optim.RMSprop(discreminator.parameters(), lr=0.0002)
    '''
    optimizerG = torch.optim.Adam(generator.parameters())#, lr=learningRate, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(discreminator.parameters())#, lr=learningRate, betas=(0.5, 0.999))
    '''
    optimizerG = torch.optim.Adadelta(generator.parameters(), lr=0.01)#TODO
    optimizerD = torch.optim.Adadelta(discreminator.parameters(), lr=0.01)#TODO
    '''
    for ep in range(EPOCHS):
        miss = 0.0
        fool = 0.0
        ld = 0.0
        lg = 0.0
        for i, data in enumerate(dataloader, 0):
#TRAIND
            #T
            for _ in range(DBias):
                optimizerD.zero_grad()
                x_real = Variable(data)
                labels = torch.ones((BATCHSIZE,1))
                labels = Variable(labels, requires_grad=False).double()
                if Iscuda:
                    x_real = x_real.cuda()
                    labels = labels.cuda()
                y = discreminator(x_real)
                real_score = y.mean()
                
                lossD_r = binCE(y, labels)
                #lossD_r = -torch.mean((y))
                
                if Iscuda:
                    lossD_r = lossD_r.cuda()    
                lossD_r.backward(retain_graph=True)
                #F
                noise = torch.randn(BATCHSIZE, ChannelEnc).double()
                if Iscuda:
                    noise = noise.cuda()
                labels.fill_(0.)
                x_fake = generator(noise)
                y = discreminator(x_fake.detach())
                fake_score = y.mean()
                
                lossD_f = binCE(y, labels)
                #lossD_f = torch.mean((y))
                
                if Iscuda:
                    lossD_f = lossD_f.cuda()
                lossD_f.backward(retain_graph=True)
                #lossD = lossD_f - lossD_r
                lossD = (lossD_f + lossD_r)/2
                
                optimizerD.step()
#TRAING
            for _ in range(GBias):
                optimizerG.zero_grad()
                labels.fill_(1.)
                y = discreminator(x_fake)
                fool_score = y.mean()
                
                lossG = binCE(y, labels)
                #-torch.mean((y))
                
                if Iscuda:
                    lossG = lossG.cuda()
                lossG.backward(retain_graph=True)
                optimizerG.step()
#PRINT
            miss += real_score.item()
            #miss += fake_score.data[0]
            fool += fool_score.item()
            ld += lossD.item()
            lg += lossG.item()
            if i%PRINTSIZE==PRINTSIZE - 1:
                loss_log[0].append(miss/PRINTSIZE)
                loss_log[1].append(fool/PRINTSIZE)
                loss_log[2].append(ld/PRINTSIZE)
                loss_log[3].append(lg/PRINTSIZE)
                process_bar.show_process(miss/PRINTSIZE*100, fool/PRINTSIZE*100, ld/PRINTSIZE, lg/PRINTSIZE, ep, i)
                miss, fool, ld, lg  = (0,0,0,0)
        start_i = int(ep*numSteps/PRINTSIZE)
        end_i = int((ep + 1)*numSteps/PRINTSIZE)
        miss = np.mean(np.array(loss_log[0][start_i:end_i]).astype(float))
        fool = np.mean(np.array(loss_log[1][start_i:end_i]).astype(float))
        ld = np.mean(np.array(loss_log[2][start_i:end_i]).astype(float))
        lg = np.mean(np.array(loss_log[3][start_i:end_i]).astype(float))
        print(ep+1, miss, fool, ld, lg)
    
        x_step = range(0, len(loss_log[2])*PRINTSIZE*BATCHSIZE, PRINTSIZE*BATCHSIZE)
        plt.plot(x_step, loss_log[0])
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.savefig(str(VERSION)+str(Langda)+'hit.png')
        plt.close('all')
        plt.plot(x_step, loss_log[0])
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.savefig(str(VERSION)+str(Langda)+'fool.png')
        plt.close('all')
        plt.plot(x_step, loss_log[2])
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.savefig(str(VERSION)+str(Langda)+'lossG.png')
        plt.close('all')
        plt.plot(x_step, loss_log[3])
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.savefig(str(VERSION)+str(Langda)+'lossD.png')
        plt.close('all')
    
#Validation
        noise = torch.randn(BATCHSIZE, ChannelEnc).double()
        if Iscuda:
            noise = noise.cuda()
        x_fake_gpu = generator(noise)
        x_fake = x_fake_gpu.cpu()
        pred = x_fake.data.numpy()
        for i in range(BATCHSIZE):
            image = np.array(pred[i]*256.).astype(int)
            mpimg.imsave("/home/jay/hw4/pred/GAN/EP"+str(ep) +'_'+ str(i).zfill(4)+'.png', image)
        
        if ep%3 == 2:
            torch.save(generator.state_dict(), save_path+str(VERSION)+'_'+'G'+'params.pkl')
            torch.save(discreminator.state_dict(), save_path+str(VERSION)+'_'+'D'+'params.pkl')
        
    torch.save(generator.state_dict(), save_path+str(VERSION)+str(EPOCHS)+'_'+'G'+'params.pkl')
    torch.save(discreminator.state_dict(), save_path+str(VERSION)+str(EPOCHS)+'_'+'D'+'params.pkl')

if __name__ == "__main__":
    main()