#!/bin/bash
wget https://www.dropbox.com/s/2s1spyy83w4y5yt/FCNw22.hdf5?dl=1

mv FCN23.hdf5?dl=1 FCN23.hdf5

python3 test.py $1 $2