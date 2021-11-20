"""
https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
#
import tensorflow as tf
from keras import Model, layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

t0 = time.time()

# ===================================================
Rician_factor = 5
power_B = 0.05  # Watts
M_Bobs = 4
power_E = 0.03  # Watts
N_Eves = 2
n_anten_Alice = 15
# RIS
K_RIS = 8
phi_list = [np.pi for i in range(K_RIS)]
phi_array = np.array(phi_list)
Phi_RIS = np.diag(np.exp(1j*2*np.pi*phi_array))
xi = 0.1  # temporal-correlation coefficient

# ===================================================
""" Fit a model & Predict the future """
# def LSTM_AE__sequential_model(trainX_3D, testX_3D):
#     epochs, batch_size = 500, 32
#     n_features = trainX_3D.shape[2]
#     ''' define model using the Sequential API '''
#     model = Sequential()
#     model.add(LSTM(64, activation='relu', return_sequences=True))
#     model.add(LSTM(32, activation='relu', return_sequences=True))
#     model.add(LSTM(16, activation='relu', return_sequences=True))
#     model.add(LSTM(32, activation='relu', return_sequences=True))
#     model.add(LSTM(64, activation='relu', return_sequences=True))
#     model.add(TimeDistributed(Dense(n_features)))
#     model.compile(optimizer='adam', loss='mse')
#     ''' fit model '''
#     model.fit(trainX_3D, trainX_3D,
#               epochs=epochs, batch_size=batch_size, verbose=0)
#     testX_hat_3D = model.predict(testX_3D, verbose=0)
#     return testX_hat_3D


def LSTM_AE__functional_model(trainX_3D):
    epochs, batch_size = 70, 32
    n_time_steps_IN = trainX_3D.shape[1]
    n_features = trainX_3D.shape[2]
    ''' define model using the functional API '''
    input_ = layers.Input(shape=(n_time_steps_IN, n_features))
    hidden = LSTM(64, activation='relu', return_sequences=True)(input_)
    hidden = LSTM(32, activation='relu', return_sequences=True)(hidden)
    hidden = LSTM(16, activation='relu', return_sequences=True)(hidden)
    hidden = LSTM(32, activation='relu', return_sequences=True)(hidden)
    hidden = LSTM(64, activation='relu', return_sequences=True)(hidden)
    output_ = TimeDistributed(Dense(n_features))(hidden)
    model = Model(input=input_, output=output_)
    model.compile(optimizer='adam', loss='mse')
    ''' fit model '''
    history = model.fit(trainX_3D, trainX_3D,
                        epochs=epochs, batch_size=batch_size, verbose=0)
    trainX_hat_3D = model.predict(trainX_3D, verbose=0)
    return model, history, trainX_hat_3D


# ===================================================
n_time_steps_IN = 15
n_steps_OUT = 5
""" Create 'folder_name' in 'input' """
input_folder = '/input/NA' + str(n_anten_Alice)\
                + '_B' + str(power_B)\
                + '_E' + str(power_E)\
                + '_vB' + str(60)\
                + '_NE' + str(N_Eves)\
                + '_K' + str(Rician_factor)\
                + '_IN' + str(n_time_steps_IN)\
                + '_OUT' + str(n_steps_OUT)
input_path = os.path.abspath(os.getcwd()) + input_folder
path_to_input = os.path.join(input_path, '')
if not os.path.exists(path_to_input):  # check if the subfolder exists
    os.makedirs(path_to_input)  # create the subfolder


""" Loading data """
trainX_3D_H0 = np.load(input_path+'/trainX_H0__LSTM_IN.npy')
testX_3D_H0 = np.load(input_path+'/testX_H0__LSTM_IN.npy')
testX_3D_H1 = np.load(input_path+'/testX_H1__LSTM_IN.npy')
testX_3D = np.vstack((testX_3D_H0, testX_3D_H1))
n_samples_train = trainX_3D_H0.shape[0]
n_samples_test_H0 = testX_3D_H0.shape[0]
n_samples_test_H1 = testX_3D_H1.shape[0]
n_samples_test = n_samples_test_H0 + n_samples_test_H1
n_features = trainX_3D_H0.shape[2]
print('n_time_steps_IN =', n_time_steps_IN)

cur_sample_i = 0
cur_anten_ell = 0
time_indices_for_a_sample = [(i+1) for i in range(n_time_steps_IN)]
plt.figure()
plt.plot(time_indices_for_a_sample,
         testX_3D_H0[cur_sample_i, :, cur_anten_ell],
         color='b')
plt.plot(time_indices_for_a_sample,
         testX_3D_H1[cur_sample_i, :, cur_anten_ell],
         color='r', linestyle='--')
plt.xlim((0, n_time_steps_IN))
plt.xlabel(r'$1\leq \mu\leq\Delta = %s$' % (n_time_steps_IN), fontsize=12)
plt.ylabel(r'$q^{[\mu]}_{%s}$' % (cur_anten_ell+1), fontsize=15)
plt.show()

# ===================================================
""" Illustration """
sample_i = 1
fig = plt.figure()
plt.plot(testX_3D_H0[:, sample_i, 0], 'k', label=r'$H_0$ (No attack)')
plt.plot(testX_3D_H1[:, sample_i, 0], 'r', label=r'$H_1$ (Under attack)')
plt.xlabel('Time index', fontsize=12)
plt.ylabel('Signal strength', fontsize=12)
plt.xlim((0, 82))
plt.legend(loc='best')
