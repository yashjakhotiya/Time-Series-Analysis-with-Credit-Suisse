import tensorflow as tf 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from hyperparams import Hyperparams

H = Hyperparams()
    
class Model(object):

    def __init__(self):
        return None

    def get_model(self):
        look_back_limit = H.look_back_limit
        model = Sequential()
        model.add(LSTM(units=100, input_shape=(look_back_limit, 4)))
        # model.add(LSTM(units=75))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(16, activation = 'relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1))

        return model