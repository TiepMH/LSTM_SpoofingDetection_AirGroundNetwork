import os
import numpy as np
import matplotlib.pyplot as plt

# The 3 following lines are used for generating LaTeX symbols
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{lmodern}')

# ===================================================
Rician_factor = 5
power_B = 0.05  # Watts
M_Bobs = 4
power_E = 0.03  # Watts
N_Eves = 3
n_anten_Alice = 6
# RIS
K_RIS = 8
phi_list = [np.pi for i in range(K_RIS)]
phi_array = np.array(phi_list)
Phi_RIS = np.diag(np.exp(1j*2*np.pi*phi_array))
xi = 0.1  # temporal-correlation coefficient

# ===================================================
n_time_steps_IN = 15
n_steps_OUT = 5
scaling_list = [0.5+0.1*i for i in range(27)]

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

""" Load the results """
# np.save(result_path+'/loss_vs_epochs.npy', hist.history['loss'])
acc_H0_mean_arr = np.load(result_path+'/acc_H0_mean_arr.npy')
acc_H1_mean_arr = np.load(result_path+'/acc_H1_mean_arr.npy')
acc_H0_error_arr = np.load(result_path+'/acc_H0_error_arr.npy')
acc_H1_error_arr = np.load(result_path+'/acc_H1_error_arr.npy')

# ===================================================
""" Illustration """
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
#plt.ylabel('Prob. of identifying $H_{\#}$ as $H_{\#}$', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.show()

""" Save the figure """
# fig.savefig('saved_figs/' + 'Acc_vs_factor_after_50runs.png', dpi=300)
