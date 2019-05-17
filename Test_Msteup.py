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



def reshape_y_hat(y_hat,dim):
    re_y = []
    i = 0
    while i < len(y_hat):
        tmp = []
        for j in range(dim):
            tmp.append(y_hat[i+j])
        i = i + dim
        re_y.append(tmp)
    re_y = np.array(re_y,dtype='float64')
    return  re_y

#多维反归一化
def FNormalizeMult(data,normalize):

    data = np.array(data,dtype='float64')
    #列
    for i in  range(0,data.shape[1]):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        #行
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow

    return data

#使用训练数据的归一化
def NormalizeMultUseData(data,normalize):

    for i in range(0, data.shape[1]):

        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow

        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta

    return  data


#仅对最后200条数据进行测试 因为预测仅最新有作用
data = np.zeros(400)
data.dtype = 'float64'
data = data.reshape(200,2)

sinx=np.arange(0,8*np.pi,2*np.pi/50,dtype='float64')
siny=np.sin(sinx)

cosx=np.arange(0,8*np.pi,2*np.pi/50,dtype='float64')
cosy=np.cos(sinx)

data[:,0] = siny
data[:,1] = cosy


#归一化
normalize = np.load("./MultiSteup2.npy")
data = NormalizeMultUseData(data, normalize)
model = load_model("./MultiSteup2.h5")
test_X = data.reshape(1,data.shape[0],data.shape[1])
y_hat  =  model.predict(test_X)
#重组
y_hat = y_hat.reshape(y_hat.shape[1])
y_hat = reshape_y_hat(y_hat,2)

#反归一化
y_hat = FNormalizeMult(y_hat, normalize)

print(y_hat.shape)
plt.plot(y_hat[:,0])
plt.show()
plt.plot(y_hat[:,1])
plt.show()