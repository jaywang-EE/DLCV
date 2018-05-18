import numpy as np
import scipy
import matplotlib.image as mpimg
from random import randint
import sys
#import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from sklearn.manifold import TSNE



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
    plt.savefig(predPath + 'latent_tsne.png')
    #plt.show()

def TryImage(filename):
    try:
        img = mpimg.imread(filename)
        #print(filename+' is real')
        return True
    except:
        print("ERROR", filename)
        return False

def tsne(X, n_components):
    model = TSNE(n_components=2, perplexity=40)
    return model.fit_transform(X)