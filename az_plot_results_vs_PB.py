import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# The 3 following lines are used for generating LaTeX symbols
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{lmodern}')

# ===================================================
Rician_factor = 5
power_B_list = [0.03, 0.05, 0.1]  # Watts
M_Bobs = 4
power_E = 0.03  # Watts
N_Eves = M_Bobs - 2
n_anten_Alice = 6
# RIS
K_RIS = 8
phi_list = [np.pi for i in range(K_RIS)]
phi_array = np.array(phi_list)
Phi_RIS = np.diag(np.exp(1j*2*np.pi*phi_array))
xi = 0.1  # temporal-correlation coefficient

# ===================================================
linestyles = [':', '-.', '-', '--', ':']
linewidths = [2, 1.75, 1.5, 1.25, 1, 0.75]
markers = ['s', 'o', '+', '>', 'x']
colors = ['k', 'b', 'r', 'g', 'm']
fig = plt.figure()
for i in range(len(power_B_list)):
    power_B = power_B_list[i]
    n_time_steps_IN = 15
    n_steps_OUT = 5
    """ Create the subfolder 'folder_name' in the folder 'input' """
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
    testX_hat_3D_H0 = np.load(input_path+'/testX_H0__LSTM_OUT.npy')
    testX_hat_3D_H1 = np.load(input_path+'/testX_H1__LSTM_OUT.npy')
    testX_3D = np.vstack((testX_3D_H0, testX_3D_H1))
    print('n_time_steps_IN =', n_time_steps_IN)
    # ========================
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
    """ Load and depict results """
    acc_H0_mean_arr = np.load(result_path+'/acc_H0_mean_arr.npy')
    acc_H1_mean_arr = np.load(result_path+'/acc_H1_mean_arr.npy')
    acc_H0_error_arr = np.load(result_path+'/acc_H0_error_arr.npy')
    acc_H1_error_arr = np.load(result_path+'/acc_H1_error_arr.npy')
    # ========================
    """ Sensitivity = TPR, Specificity = TNR """
    TPR_mean_arr = acc_H0_mean_arr
    TNR_mean_arr = acc_H1_mean_arr
    False_Alarm_mean_arr = 1 - TNR_mean_arr
    roc_auc = auc(False_Alarm_mean_arr, TPR_mean_arr)
    # ========================
    """ Illustration """
    plt.plot(False_Alarm_mean_arr, TPR_mean_arr,
             linestyle=linestyles[i], marker='',
             color=colors[i],
             label='$P_B =$ %(first).0f mW'
             % {"first": power_B*1000})
plt.ylabel(r'Prob. of correct $(H_0)$ detection', fontsize=12)
plt.xlabel(r'Prob. of false $(H_1)$ alarm', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.xlim((-0.005, 1))
plt.ylim((-0.005, 1))
plt.axes().set_aspect('equal')
plt.tight_layout()
plt.show()

""" Save figures """
# fig.savefig('saved_figs/' + 'ROC_vs_PB.png', dpi=300)

# ===================================================
""" Illustration """
scaling_list = [0.5+0.1*i for i in range(27)]
fig, ax = plt.subplots(1, 2, figsize=(7, 4))
for i in range(len(power_B_list)):
    power_B = power_B_list[i]
    n_time_steps_IN = 15
    n_steps_OUT = 5
    """ Create the subfolder 'folder_name' in the folder 'input' """
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
    testX_hat_3D_H0 = np.load(input_path+'/testX_H0__LSTM_OUT.npy')
    testX_hat_3D_H1 = np.load(input_path+'/testX_H1__LSTM_OUT.npy')
    testX_3D = np.vstack((testX_3D_H0, testX_3D_H1))
    print('n_time_steps_IN =', n_time_steps_IN)
    # ========================
    """ Create the subfolder 'folder_name' in the folder 'results' """
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
    """ Loading results """
    acc_H0_mean_arr = np.load(result_path+'/acc_H0_mean_arr.npy')
    acc_H1_mean_arr = np.load(result_path+'/acc_H1_mean_arr.npy')
    acc_H0_error_arr = np.load(result_path+'/acc_H0_error_arr.npy')
    acc_H1_error_arr = np.load(result_path+'/acc_H1_error_arr.npy')
    # ========================
    acc_mean_arr = 0.5*(acc_H0_mean_arr + acc_H1_mean_arr)
    acc_error_arr = 0.5*(acc_H0_error_arr + acc_H1_error_arr)
    ax[0].plot(scaling_list, acc_H0_mean_arr,
               linestyle=linestyles[i], marker=markers[i],
               markerfacecolor='none',
               linewidth=linewidths[i], color=colors[i],
               label='$P_B =$ %(first).0f mW' % {"first": power_B*1000})
    ax[1].plot(scaling_list, acc_H1_mean_arr,
               linestyle=linestyles[i], marker='',
               markerfacecolor='none',
               linewidth=linewidths[i], color=colors[i],
               label='$P_B =$ %(first).0f mW' % {"first": power_B*1000})
    # plt.fill_between(scaling_list,
    #                  acc_H0_mean_arr - acc_H0_error_arr,
    #                  acc_H0_mean_arr + acc_H0_error_arr,
    #                  alpha=0.5, edgecolor='#CC4F1B', facecolor='blue',
    #                  label='')
    # plt.fill_between(scaling_list,
    #                  acc_H1_mean_arr - acc_H1_error_arr,
    #                  acc_H1_mean_arr + acc_H1_error_arr,
    #                  alpha=0.5, edgecolor='#CC4F1B', facecolor='orange',
    #                  label='')
    ax[0].set_xlabel('Threshold-determining factor', fontsize=12)
    ax[1].set_xlabel('Threshold-determining factor', fontsize=12)
    ax[0].set_ylabel(r'Prob. of correct $(H_0)$ detection', fontsize=12)
    ax[1].set_ylabel(r'Prob. of correct $(H_1)$ detection', fontsize=12)
    ax[0].legend(loc='lower right')
    ax[1].legend(loc='lower left')
    ax[0].set_xlim((scaling_list[0], scaling_list[-1]))
    ax[1].set_xlim((scaling_list[0], scaling_list[-1]))
    ax[0].set_ylim((-0.005, 1.005))
    ax[1].set_ylim((-0.005, 1.005))
plt.legend(loc='best')
fig.tight_layout()
plt.show()

""" Save figures """
# fig.savefig('saved_figs/' + 'TPR_and_TNR_vs_factor_and_PB.png', dpi=300)

# ===================================================
""" Illustration: Acc """
scaling_list = [0.5+0.1*i for i in range(27)]
fig = plt.figure(figsize=(7, 5))
for i in range(len(power_B_list)):
    power_B = power_B_list[i]
    n_time_steps_IN = 15
    n_steps_OUT = 5
    """ Create the subfolder 'folder_name' in the folder 'input' """
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
    testX_hat_3D_H0 = np.load(input_path+'/testX_H0__LSTM_OUT.npy')
    testX_hat_3D_H1 = np.load(input_path+'/testX_H1__LSTM_OUT.npy')
    testX_3D = np.vstack((testX_3D_H0, testX_3D_H1))
    print('n_time_steps_IN =', n_time_steps_IN)
    # ========================
    acc_mean_arr = 0.5*(acc_H0_mean_arr + acc_H1_mean_arr)
    acc_error_arr = 0.5*(acc_H0_error_arr + acc_H1_error_arr)
    """ Create the subfolder 'folder_name' in the folder 'results' """
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
    """ Loading results """
    acc_H0_mean_arr = np.load(result_path+'/acc_H0_mean_arr.npy')
    acc_H1_mean_arr = np.load(result_path+'/acc_H1_mean_arr.npy')
    acc_H0_error_arr = np.load(result_path+'/acc_H0_error_arr.npy')
    acc_H1_error_arr = np.load(result_path+'/acc_H1_error_arr.npy')
    # ========================
    acc_mean_arr = 0.5*(acc_H0_mean_arr + acc_H1_mean_arr)
    acc_error_arr = 0.5*(acc_H0_error_arr + acc_H1_error_arr)
    plt.plot(scaling_list, acc_mean_arr,
             linestyle=linestyles[i], marker=markers[i],
             markerfacecolor='none',
             linewidth=linewidths[i], color=colors[i],
             label='$P_B =$ %(first).0f mW' % {"first": power_B*1000})
    plt.xlabel('Threshold-determining factor', fontsize=12)
    plt.ylabel(r'Prob. of correct detection', fontsize=12)
    plt.legend(loc='lower right')
    plt.xlim((scaling_list[0], 3))
    plt.ylim((-0.005, 1))
fig.tight_layout()
plt.show()

""" Save figures """
# fig.savefig('saved_figs/' + 'Acc_vs_factor_and_PB.png', dpi=300)