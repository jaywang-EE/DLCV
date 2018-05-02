#!/bin/bash
wget https://www.dropbox.com/s/wq30pw67d024g7k/FCNw25.hdf5?dl=1

mv FCNw25.hdf5?dl=1 FCNw25.hdf5

python3 testbest.py $1 $2