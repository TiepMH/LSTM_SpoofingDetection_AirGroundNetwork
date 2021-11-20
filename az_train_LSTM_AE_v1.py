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


""" Training and testing datasets to be reconstructed """
n_evaluations = 50
scaling_list = [0.5+0.1*i for i in range(27)]
acc_H0_arr = np.zeros([n_evaluations, len(scaling_list)])
acc_H1_arr = np.zeros([n_evaluations, len(scaling_list)])
for u in range(n_evaluations):
    # name = 'aff'
    model, hist, trainX_hat_3D_H0 = LSTM_AE__functional_model(trainX_3D_H0)
    testX_hat_3D = model.predict(testX_3D, verbose=0)
    testX_hat_3D_H0 = testX_hat_3D[:n_samples_test_H0]
    testX_hat_3D_H1 = testX_hat_3D[n_samples_test_H0:]

    """ Detection Algorithm """
    diff_test_3D_H0 = abs(testX_hat_3D_H0 - testX_3D_H0)
    mean_diff_test_H0 = np.empty([0, 1])
    for n in range(n_samples_test_H0):
        mean_diff_test_H0 = np.append(mean_diff_test_H0,
                                      np.mean(diff_test_3D_H0[n, :, :]))
    # end of 'for' loop
    diff_test_3D_H1 = abs(testX_hat_3D_H1 - testX_3D_H1)
    mean_diff_test_H1 = np.empty([0, 1])
    for n in range(n_samples_test_H1):
        mean_diff_test_H1 = np.append(mean_diff_test_H1,
                                      np.mean(diff_test_3D_H1[n, :, :]))
    # end of 'for' loop

    """ Detection Threshold """
    diff_train_3D_H0 = abs(trainX_hat_3D_H0 - trainX_3D_H0)
    thrsh = np.mean(diff_train_3D_H0)

    """ Calculate the probabilities """
    acc_H0_list = []
    acc_H1_list = []
    for scaling in scaling_list:
        thrsh_scaled = scaling*thrsh
        H0_decoded_as_H0 = (mean_diff_test_H0 <= thrsh_scaled)*1
        H1_decoded_as_H1 = (mean_diff_test_H1 > thrsh_scaled)*1
        acc_H0 = np.sum(H0_decoded_as_H0)/len(H0_decoded_as_H0)
        acc_H1 = np.sum(H1_decoded_as_H1)/len(H1_decoded_as_H1)
        acc_H0_list.append(acc_H0)
        acc_H1_list.append(acc_H1)
    # end of 'for' loop
    acc_H0_arr[u, :] = np.array(acc_H0_list)
    acc_H1_arr[u, :] = np.array(acc_H1_list)

# ===================================================
""" Create 'folder_name' in 'output' """
# output_folder = '/output/NA' + str(n_anten_Alice)\
#                 + '_B' + str(power_B)\
#                 + '_E' + str(power_E)\
#                 + '_vB' + str(60)\
#                 + '_NE' + str(N_Eves)\
#                 + '_K' + str(Rician_factor)\
#                 + '_IN' + str(n_time_steps_IN)\
#                 + '_OUT' + str(n_steps_OUT)
# output_path = os.path.abspath(os.getcwd()) + output_folder
# path_to_output = os.path.join(output_path, '')
# if not os.path.exists(path_to_output):  # check if the subfolder exists
#     os.makedirs(path_to_output)  # create the subfolder

""" Save the OUTPUT of LSTM """
# np.save(output_path+'/trainX_H0__LSTM_OUT.npy', trainX_hat_3D_H0)
# np.save(output_path+'/testX_H0__LSTM_OUT.npy', testX_hat_3D_H0)
# np.save(output_path+'/testX_H1__LSTM_OUT.npy', testX_hat_3D_H1)

# ===================================================
acc_H0_mean_arr = np.mean(acc_H0_arr, axis=0)
acc_H1_mean_arr = np.mean(acc_H1_arr, axis=0)
acc_H0_error_arr = np.sqrt(np.var(acc_H0_arr, axis=0))
acc_H1_error_arr = np.sqrt(np.var(acc_H1_arr, axis=0))
# ===================================================
""" Create the subfolder 'folder_name' in the folder 'input' """
result_folder = '/results/NA' + str(n_anten_Alice)\
                + '_B' + str(power_B)\
                + '_E' + str(power_E)\
                + '_vB' + str(60)\
                + '_NE' + str(N_Eves)\
                + '_K' + str(Rician_factor)\
                + '_IN' + str(n_time_steps_IN)\
                + '_OUT' + str(n_steps_OUT)
result_path = os.path.abspath(os.getcwd()) + result_folder
path_to_results = os.path.join(result_path, '')
if not os.path.exists(path_to_results):  # check if the subfolder exists
    os.makedirs(path_to_results)  # create the results

""" Save the results """
# np.save(result_path+'/loss_vs_epochs.npy', hist.history['loss'])
# np.save(result_path+'/acc_H0_mean_arr.npy', acc_H0_mean_arr)
# np.save(result_path+'/acc_H1_mean_arr.npy', acc_H1_mean_arr)
# np.save(result_path+'/acc_H0_error_arr.npy', acc_H0_error_arr)
# np.save(result_path+'/acc_H1_error_arr.npy', acc_H1_error_arr)

# ===================================================
t1 = time.time()

print('Run time =', np.round((t1-t0)/60, 2), 'mins')

# The 3 following lines are used for generating LaTeX symbols
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{lmodern}')

""" Illustration """
plt.figure()
plt.plot(hist.history['loss'])
plt.ylim(ymax=20000)
plt.show()

fig = plt.figure()
plt.plot(scaling_list, acc_H0_mean_arr, 'b',
         label=r'Prob. of identifying $H_0$ as $H_0$')
plt.plot(scaling_list, acc_H1_mean_arr, 'r--',
         label=r'Prob. of identifying $H_1$ as $H_1$')
plt.fill_between(scaling_list,
                 acc_H0_mean_arr - acc_H0_error_arr,
                 acc_H0_mean_arr + acc_H0_error_arr,
                 alpha=0.5, edgecolor='#CC4F1B', facecolor='blue')
plt.fill_between(scaling_list,
                 acc_H1_mean_arr - acc_H1_error_arr,
                 acc_H1_mean_arr + acc_H1_error_arr,
                 alpha=0.5, edgecolor='#CC4F1B', facecolor='orange')
plt.xlim((0.5, 3))
plt.ylim((0, 1))
plt.legend(loc='best', fontsize=12)
plt.xlabel('Threshold-determining factor', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.show()

# fig.savefig('saved_figs/' + 'Acc_vs_factor_after_50runs.png', dpi=300)


