import os
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.1
fdr = 0.05
n_clusters = 500
n_methods = 5
draws = 50
# n_sim = 50
# n_params = 7
n_experiments = 4

n_params_dict = {'snr': 5, 'rho': 10, 'n': 7, 'sparsity': 5}
labels_dict = {'snr': 'Signal-to-noise ratio', 'rho': r'$\rho$', 'n': 'n_samples', 'sparsity': 'sparsity'}
order_dict = {'H-mean': 0, 'e-values': 1, 'AKO': 2, 'Vanilla KO': 3, 'Closed testing': 4}
expe_order_dict = {0: 'rho', 1: 'snr', 2: 'n', 3: 'sparsity'}
n_sims_dict = {'snr': 50, 'rho': 50, 'n': 50, 'sparsity': 50}

def draw_single_curve(param, ax, n_params_dict, n_sim, n_methods, draws, alpha, fdr, use_fdp):
    n_params = n_params_dict[param]
    curves_tot = np.zeros((n_params, n_methods, n_sim))
    sizes_tot = np.zeros((n_params, n_methods, n_sim))
    path_to_results = f'../figures/{param}/alpha{alpha}/fdr{fdr}'

    compt = 0
    sorter_ = []
    sorter__ = []
    for path in os.listdir(path_to_results):
        if 'pdf' not in path and '2000' in path and 'k_max' in path and 'storing_sizes' in path:
            bounds = np.load(os.path.join(path_to_results, path), allow_pickle=True)
            path_sizes = path.replace("_storing","")
            sizes = np.load(os.path.join(path_to_results, path_sizes), allow_pickle=True)
            bounds_fdp = bounds[:n_methods]
            runs = len(bounds_fdp[0])
            bounds_tdp = bounds[n_methods:]
            if use_fdp:
                curves_tot[compt] = bounds_fdp
            else:
                curves_tot[compt] = bounds_tdp

            sizes_tot[compt] = sizes
    
            compt = compt + 1

            param_current = float(path.rsplit(param, 6)[1].rsplit('_', 12)[0])
            sorter_.append(param_current)


    sorter = np.argsort(sorter_)
    param_list = np.sort(sorter_)

    curves_tot = curves_tot[sorter]
    
    if use_fdp:
        var_band = 1.96 * np.sqrt(alpha * (1 - alpha) / n_sim)
        ax.hlines(alpha, xmin=min(param_list), xmax=max(param_list), color='red', linewidth=4)
        ax.annotate(r'$\alpha$', (- 0.08, (alpha / 0.4) - 0.01), fontsize=25, color='red', xycoords='axes fraction')
        ax.fill_between(param_list, alpha, alpha + var_band, alpha=0.2, color='red')
        ax.set_ylim(-0.01, 0.4)
        ax.set_ylabel('1 - coverage', fontsize=19)
        len1, len2 = curves_tot.shape[0], curves_tot.shape[1]
        mean_ = np.ones((len1, len2))
        for j in range(len1):
            for l in range(len2):
                mean_[j][l] = len(np.where(curves_tot[j][l] > fdr)[0]) / n_sim
                # Count violations of guarantee
    
    else:
        ax.set_ylabel('Empirical power', fontsize=19)
        mean_ = np.mean(curves_tot, axis=2)

    ax.tick_params(labelsize=19)

    ax.set_xlabel(labels_dict[param], fontsize=19)

    err_mat = np.zeros((n_methods, len(param_list)))
    for method in range(n_methods):
        for l in range(len(param_list)):
            current = curves_tot[l, method]
            err_mat[method][l] = 1.96 * (current.std() / np.sqrt(len(current)))
    
    ax.plot(param_list, mean_[:, 0], label='KOPI',
                        color='blue')
    ax.plot(param_list, mean_[:, 1], label='Closed Testing',
                        color='green', linewidth=1.3)
    ax.plot(param_list, mean_[:, 2], label='e-values',
                        color='m', linewidth=1.3)
    ax.plot(param_list, mean_[:, 3], label='AKO',
                        color='c')
    ax.plot(param_list, mean_[:, 4], label='Vanilla KO',
                        color='grey')

    if not use_fdp:                     
        ax.fill_between(param_list, mean_[:, 0] - err_mat[0], mean_[:, 0] + err_mat[0], alpha=0.2, color='blue')
        ax.fill_between(param_list, mean_[:, 1] - err_mat[1], mean_[:, 1] + err_mat[1], alpha=0.2, color='green')
        ax.fill_between(param_list, mean_[:, 2] - err_mat[2], mean_[:, 2] + err_mat[2], alpha=0.2, color='m')
        ax.fill_between(param_list, mean_[:, 3] - err_mat[3], mean_[:, 3] + err_mat[3], alpha=0.2, color='c')
        ax.fill_between(param_list, mean_[:, 4] - err_mat[4], mean_[:, 4] + err_mat[4], alpha=0.2, color='grey')


fig, axs = plt.subplots(n_experiments, 2, figsize=(12, 12))
fig.subplots_adjust(left=0.1, bottom=0.1, right=None,
                    top=None, wspace=0.3, hspace=0.5)
for i in range(n_experiments):
    for j in range(2):
        use_fdp = (j == 0)
        draw_single_curve(expe_order_dict[i], axs[i][j], n_params_dict, n_sims_dict[expe_order_dict[i]], n_methods, draws, alpha, fdr, use_fdp=use_fdp)

handles, labels = axs[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, prop={'size': 19})

path_to_figs = '../figures'
plt.savefig(os.path.join(path_to_figs, 'figure2.pdf'))
plt.show()
