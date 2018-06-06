import numpy as np
import scipy
import matplotlib.image as mpimg
from random import randint
import sys

import skvideo.io
import skimage.transform
import csv
import collections
import os
from params import *
#READFILE
from os.path import isfile, isdir, join, splitext

def images2npy(path, Ntrain, IsTest = False):
    file_list = [file for file in os.listdir(path) if file.endswith('.jpg')]
    file_list.sort()

    trainSize = len(file_list)
    if Ntrain > 0 and Ntrain < trainSize:
        trainSize = Ntrain
    
    x_train = np.zeros((trainSize, 240, 320, 3))
    for i, file in enumerate(file_list):
        #print(file)
        if Ntrain == i:
            break
        fileAddr = join(path,file)
        try:
            img = mpimg.imread(fileAddr)
            x_train[i] = img
        except:
            assert False,("ERROR "+fileAddr)

        if (i+1)%1000 == 0:
            print("LOADING:",path,i)

    x_train = centre_crop(x_train) if IsTest else random_crop(x_train)
    return x_train

def label2rgb(label):
    colorMaps = [(  0,  0,  0), ( 65,105,225), (176,196,222), (255,255,255),\
                 (255,246,143), (238,238,  0), ( 34,139, 34), (205, 92, 92),\
                 (178, 34, 34), (199, 21,133), (205, 41,144)]
    return colorMaps[label]
    
    
def calSubACC(label1, label2):
    assert label1.shape[0] == label2.shape[0]
    acc = 0.
    for i in range(label1.shape[0]):
        if label1[i]==label2[i]:
            acc+=1.
    acc/=label1.shape[0]
    return acc
    
def halfFilm(arr2D):
    arr = arr2D[::2]
    shape = arr.shape
    if shape[0] > maxTime:
        arr = halfFilm(arr)
    return arr

def random_crop(arr4D,target_length=224):
    assert target_length < ROW
    assert target_length < COL
    w = randint(0,ROW - target_length)
    h = randint(0,COL - target_length)
    return(arr4D[:, w:w+target_length, h:h+target_length])
    
def centre_crop(arr4D,target_length=224):
    assert target_length < ROW
    assert target_length < COL
    w = int((ROW - target_length)/2)
    h = int((COL - target_length)/2)
    return(arr4D[:, w:w+target_length, h:h+target_length])
    

def generate_batch_data_random(x, y, batch_size):
    ylen = y.shape[0]
    loopcount = ylen // batch_size
    while (True):
        i = randint(0,loopcount-1)
        yield x[i : batch_size*loopcount : loopcount], y[i : batch_size*loopcount : loopcount]
#        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]

def int2OH(array):
    retArr = np.zeros((array.shape[0], nClass))
    retArr[np.arange(array.shape[0]), array] = 1.
    return retArr

def readShortVideo(video_path, video_category, video_name, downsample_factor=12, rescale_factor=1):
    '''
    @param video_path: video directory
    @param video_category: video category (see csv files)
    @param video_name: video name (unique, see csv files)
    @param downsample_factor: number of frames between each sampled frame (e.g., downsample_factor = 12 equals 2fps)
    @param rescale_factor: float of scale factor (rescale the image if you want to reduce computations)

    @return: (T, H, W, 3) ndarray, T indicates total sampled frames, H and W is heights and widths
    '''

    filepath = video_path + '/' + video_category
    filename = [file for file in os.listdir(filepath) if file.startswith(video_name)]
    video = os.path.join(filepath,filename[0])

    videogen = skvideo.io.vreader(video)
    frames = []
    for frameIdx, frame in enumerate(videogen):
        if frameIdx % downsample_factor == 0:
            frame = skimage.transform.rescale(frame, rescale_factor, mode='constant', preserve_range=True).astype(np.uint8)
            frames.append(frame)
        else:
            continue

    return np.array(frames).astype(np.uint8)


def getVideoList(data_path):
    '''
    @param data_path: ground-truth file path (csv files)

    @return: ordered dictionary of videos and labels {'Action_labels', 'Nouns', 'End_times', 'Start_times', 'Video_category', 'Video_index', 'Video_name'}
    '''
    result = {}

    with open (data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column, value in row.items():
                result.setdefault(column,[]).append(value)

    od = collections.OrderedDict(sorted(result.items()))
    return od
