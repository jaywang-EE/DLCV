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
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
    
def tsne(X, n_components):
    model = TSNE(n_components=2, perplexity=40)
    return model.fit_transform(X)

def plot_scatter(x, labels, title, predPath, txt=False):
    plt.title(title)
    ax = plt.subplot()
    ax.scatter(x[:,0], x[:,1], c = labels)
    txts = []
    if txt:
        for i in range(10):
            xtext, ytext = np.median(x[labels == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    plt.savefig(predPath)