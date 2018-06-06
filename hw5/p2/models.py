from params import *
import random
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Embedding, LSTM, Bidirectional

def LSTMBi(inputShape):
    model = Sequential()
    model.add(Bidirectional(LSTM(input_shape=(inputShape[0], inputShape[1]),\
            units=256, init='uniform',\
            inner_init='uniform', forget_bias_init='one', return_sequences=True,\
            activation='tanh', inner_activation='sigmoid'), \
            input_shape=(inputShape[0], inputShape[1]),name='BiLSTM'))
    model.add(Dropout(0.5))
    model.add(Dense(nClass, activation='sigmoid',name='classifier'))#CATOUTPUT
    return model