from models import *
import torch
import torch.nn as nn
def main():

    
    predPath = (sys.argv[2])
    
    encoder = E()
    decoder = D()
    model = VAE(encoder, decoder)
    
    print(model)