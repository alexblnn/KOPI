import warnings
import sys
import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
import matplotlib
matplotlib.use('Agg')
import numpy as np
from utils import compare_all_methods

seed = 42

alpha_list = [0.05, 0.1]
fdr_list = [0.05, 0.1]

n_alphas = len(alpha_list)
n_fdr = len(fdr_list)

rho = float(sys.argv[1])
sparsity = 0.1
n_samples = 500
n_clusters = 500
k_max = int(n_clusters/50)
n_methods = 5
n_jobs = 40
B = 2000
method = 'lasso_cv'
draws = 50
snr = 2

param = 'rho'
print(f'{param} = {rho}')

fdp_matrix, tdp_matrix, size_matrix = compare_all_methods(rho, sparsity, fdr_list, 
                       alpha_list, n_methods=n_methods, n_samples=n_samples, 
                       n_clusters=n_clusters, k_max=k_max,
                       repeats=50,
                       B=B, method='lasso_cv',
                       use_same_alpha=True,
                       snr=snr,
                       draws=draws, n_jobs=n_jobs,
                       seed=seed)

for j in range(n_alphas):
        for k in range(n_fdr):
            alpha = alpha_list[j]
            fdr = fdr_list[k]
            size_current = size_matrix[j][k].T.tolist()
            bounds_fdp_current = fdp_matrix[j][k].T.tolist()
            bounds_tdp_current = tdp_matrix[j][k].T.tolist()
            bounds_tot = np.concatenate([bounds_fdp_current, bounds_tdp_current])

            np.save(f'../../figures/{param}/alpha{alpha}/fdr{fdr}/results_n{n_samples}_rho{rho}_B{B}_draws{draws}_fdr{fdr}_snr{snr}_sparsity{sparsity}_k_max{k_max}_storing_sizes.npy', bounds_tot)
            np.save(f'../../figures/{param}/alpha{alpha}/fdr{fdr}/results_n{n_samples}_rho{rho}_B{B}_draws{draws}_fdr{fdr}_snr{snr}_sparsity{sparsity}_k_max{k_max}_sizes.npy', size_current)
