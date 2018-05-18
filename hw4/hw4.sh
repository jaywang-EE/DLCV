#!/bin/bash
wget https://www.dropbox.com/s/t5pku5dyrbxqoen/GANG.pkl?dl=1
mv GANG.pkl?dl=1 GANG.pkl
wget https://www.dropbox.com/s/zz1sgy9dom2mik1/GAND.pkl?dl=1
mv GAND.pkl?dl=1 GAND.pkl
wget https://www.dropbox.com/s/yl6l9lb3j0avbal/VAE.pkl?dl=1
mv VAE.pkl?dl=1 VAE.pkl

python3 VAEt/test.py  $1 $2
python3 GAN/test.py  $1 $2