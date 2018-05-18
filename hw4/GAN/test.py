#VAE
import sys
import numpy as np
import random
import scipy
import matplotlib.image as mpimg
#DEBUG
import matplotlib.pyplot as plt
from models import *
from utils import *
#from params import *
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import sklearn
from sklearn.manifold import TSNE
import csv

IsLoad = True
load_model = VAE.pkl#"/home/jay/hw4/models/VAEt/VAE_051811_1params.pkl"

np.set_printoptions(threshold=np.nan)

def main():
    np.random.seed(326)
    torch.manual_seed(1)
    assert len(sys.argv) > 2, 'NO ENOUGH ARGV'
    
    dataPath = (sys.argv[1])
    testPath = dataPath + 'test/'
    csvPath = dataPath + 'test.csv'
    
    predPath = (sys.argv[2])
    
    encoder = E()
    decoder = D()
    model = VAE(encoder, decoder)
    if Iscuda:
        model.cuda()
    model.load_state_dict(torch.load(load_model))
    print("LOADMODEL : ", load_model)
    print('*'*20, "LOADMODEL", '*'*20)
#IMGREAD
    readThrough = True
    file_list = [file for file in os.listdir(testPath) if file.endswith('.png')]
    file_list.sort()
    
    x_train = np.zeros((500, 64, 64, 3))
    for i, file in enumerate(file_list):
        if i == 500:
            break
        fileAddr = testPath + file
        readThrough = TryImage(fileAddr) and readThrough
        if readThrough:
            x_train[i] = mpimg.imread(fileAddr)
        else:
            x_train = []
            break
        if i%1000 == 0 and i != 0:
            print("LOADING X...",i)
#CSVREAD
    f = open(csvPath, 'r', newline='')
    cls = np.zeros((500, 13))
    count = 0
    for row in csv.reader(f):
        if count == 0:
            count += 1
            continue
        if 500 < count:#STDFM1
            break
        cls[count - 1] = (np.array(row[1:]))
        count += 1
    f.close()
    cls = cls[:,9]#SMILE
    cls = cls.astype(int)
    print(cls.shape)
    
    if readThrough:
        x_train_torch = torch.from_numpy(x_train).double()
        #cls = torch.from_numpy(cls).double()
    if Iscuda:
        x_train_torch = x_train_torch.cuda()
#TEST10
    print('*'*20, "REC", '*'*20)
    print(x_train[0:10].shape)
    x_train_torch_10 = torch.from_numpy(x_train[0:10]).double()
    if Iscuda:
        x_train_torch_10 = x_train_torch_10.cuda()
    x_rec_gpu = model(x_train_torch_10)
    x_rec = x_rec_gpu.cpu()
    pred = x_rec.data.numpy()
    for i in range(10):
        image = np.array(pred[i]*256.).astype(int)
        mpimg.imsave(predPath + "VAE10_" + str(i).zfill(2)+'.png', image)
#RAND32
    print('*'*20, "RND", '*'*20)
    noise = torch.randn(32, 512).double()
    if Iscuda:
        noise = noise.cuda()
    x_fake_gpu = model.decoder(noise)
    x_fake = x_fake_gpu.cpu()
    pred = x_fake.data.numpy()
    for i in range(32):
        image = np.array(pred[i]*256.).astype(int)
        mpimg.imsave(predPath + "VAE32_" + str(i).zfill(2)+'.png', image)
#TSNE
    print('*'*20, "TSNE", '*'*20)
    y_latent_gpu = model.encoder(x_train_torch)
    y_latent = y_latent_gpu.cpu()
    latent = y_latent.data.numpy()
    
    print('*'*20, "DRAW", '*'*20)
    latent_tsne = tsne(latent, 2)
    plot_scatter(latent_tsne, cls, "latent with tsne")
        
if __name__ == "__main__":
    main()