import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import Callback
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import  pandas as pd
import  os
import  keras.callbacks
import matplotlib.pyplot as plt

#设定为自增长
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)



def create_dataset(data,n_predictions,n_next):
    '''
    对数据进行处理
    '''
    dim = data.shape[1]
    train_X, train_Y = [], []
    for i in range(data.shape[0]-n_predictions-n_next-1):
        a = data[i:(i+n_predictions),:]
        train_X.append(a)
        tempb = data[(i+n_predictions):(i+n_predictions+n_next),:]
        b = []
        for j in range(len(tempb)):
            for k in range(dim):
                b.append(tempb[j,k])
        train_Y.append(b)
    train_X = np.array(train_X,dtype='float64')
    train_Y = np.array(train_Y,dtype='float64')

    return train_X, train_Y


def NormalizeMult(data):
    '''
    归一化 适用于单维和多维
    返回归一化后的数据和最大最小值
    '''
    normalize = np.arange(2*data.shape[1],dtype='float64')
    normalize = normalize.reshape(data.shape[1],2)

    for i in range(0,data.shape[1]):

        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])

        normalize[i,0] = listlow
        normalize[i,1] = listhigh

        delta = listhigh - listlow
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta

    return  data,normalize


def trainModel(train_X, train_Y):
    '''
    trainX，trainY: 训练LSTM模型所需要的数据
    '''
    model = Sequential()
    model.add(LSTM(
        140,
        input_shape=(train_X.shape[1], train_X.shape[2]),
        return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(
        140,
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        train_Y.shape[1]))
    model.add(Activation("relu"))

    model.compile(loss='mse', optimizer='adam')
    model.fit(train_X, train_Y, epochs=100, batch_size=64, verbose=1)

    return model

#进行测试
data = np.zeros(2000)
data.dtype = 'float64'
data = data.reshape(1000,2)

sinx=np.arange(0,40*np.pi,2*np.pi/50,dtype='float64')
siny=np.sin(sinx)

cosx=np.arange(0,40*np.pi,2*np.pi/50,dtype='float64')
cosy=np.cos(sinx)

data[:,0] = siny
data[:,1] = cosy



print(data)
plt.plot(data[:,0])
plt.show()
plt.plot(data[:,1])
plt.show()

#归一化的加入

data,normalize = NormalizeMult(data)

train_X,train_Y = create_dataset(data,200,50)
model = trainModel(train_X,train_Y)

np.save("./MultiSteup2.npy",normalize)
model.save("./MultiSteup2.h5")