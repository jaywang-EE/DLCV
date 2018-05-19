#GAN
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
import csv

IsLoad = True
load_modelD = "GAND.pkl"
load_modelG = "GANG.pkl"

np.set_printoptions(threshold=np.nan)

def main():
    np.random.seed(326)
    torch.manual_seed(1)
    assert len(sys.argv) > 2, 'NO ENOUGH ARGV'
    
    dataPath = (sys.argv[1])
    testPath = dataPath + 'test/'
    csvPath = dataPath + 'test.csv'
    predPath = (sys.argv[2])
#fig2_2
    lossD = np.load('GAN/lossD.npy')
    lossG = np.load('GAN/lossG.npy')
    fool = np.load('GAN/fool.npy')
    hit = np.load('GAN/hit.npy')
    
    plt.subplot(1,2,1)
    plt.title('acc. rate')
    x_axix = list(range(len(hit)))
    plt.plot(x_axix, hit, color='green', label='Real')
    plt.plot(x_axix, fool, color='red', label='Fake')
    plt.legend()
    plt.xlabel('steps')
    plt.ylabel('rate(%)')
    #plt.savefig(predPath + 'fig2_2.jpg')
    
    plt.subplot(1,2,2)
    plt.title('training curve')
    x_axix = list(range(len(hit)))
    plt.plot(x_axix, lossD, color='green', label='D')
    plt.plot(x_axix, lossG, color='red', label='G')
    plt.legend()
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.savefig(predPath + 'fig2_2.jpg')

#fig2_3
    generator = D()
    discreminator = Discreminator(E())
    if Iscuda:
        generator.cuda()
        discreminator.cuda()
    else:
        print("ERROR: NO MODULE NAMED", structure)
    
    discreminator.load_state_dict(torch.load(load_modelD))
    generator.load_state_dict(torch.load(load_modelG))

    print("LOADMODEL : ", load_modelD)
    print("LOADMODEL : ", load_modelG)
    print('*'*20, "LOADMODEL", '*'*20)
    
    print('*'*20, "RND", '*'*20)
    plt.figure(figsize=(40,20))
    noise = torch.randn(32, 100).double()
    if Iscuda:
        noise = noise.cuda()
    x_fake_gpu = generator(noise)
    x_fake = x_fake_gpu.cpu()
    pred = x_fake.data.numpy()
    for i in range(32):
        plt.subplot(4, 8, i+1)
        image = np.array(pred[i]*256.).astype(int)
        plt.imshow(image)
    plt.axis('off')
    plt.savefig(predPath + 'fig2_3.jpg')
    plt.close()

if __name__ == "__main__":
    main()