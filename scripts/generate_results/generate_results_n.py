import os

rho_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.775, 0.85]
n_list = [300, 400, 500, 750, 1000, 1250, 1500]
sparsity_list = [0.05, 0.1, 0.15, 0.2, 0.25]
snr_list = [1, 1.5, 2, 2.5, 3]

path_to_script = ''

for n in n_list:
    os.system(path_to_script + ' python3 single_parameter_n.py ' + str(n))


