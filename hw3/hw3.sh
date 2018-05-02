#!/bin/bash
wget https://www.dropbox.com/s/tfcswcm5kga1l4p/FCN23.hdf5?dl=1

mv FCN23.hdf5?dl=1 FCN23.hdf5

python3 test.py $1 $2