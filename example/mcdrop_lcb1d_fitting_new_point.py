import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from experiments.math_functions import get_function
from pybnn import MCDROP, DNGO, Bohamiann, MCCONCRETEDROP,LCBNN
import time
from matplotlib import cm

func_name='gramcy1D'
def f(x_0):
    x = 2*x_0 + 0.5
    return (np.sin(x * 4 * np.pi) / (2*x) + (x-1)**4)-4

x_next = 0.2

n_init = 20
var_noise = 1.0e-10

seed_range = [0]
length_scale = 1e-1
T = 100
dropout = 0.05
weight_decay = 1e-6
num_epochs_set = [500,5000]
n_units_set = [10,50,100]

for num_epochs in num_epochs_set:
    for n_units in n_units_set:
        for s in seed_range:
            print(f'seed={s}')
            seed = 3
            np.random.seed(seed)
            x = np.random.rand(n_init)[:, None]
            y = f(x)
            grid = np.linspace(0, 1, 100)[:, None]
            fvals = f(grid)
            # plt.plot(x,y,'ro')
            # plt.plot(grid.flatten(),fvals,'r')
            # plt.show()
            # Add 1 new point
            y_next = f(x_next)
            x_new  = np.vstack((x, x_next))
            y_new  = np.vstack((y, y_next))

            # -- Train and Prediction with MC Dropout Model ---
            model_mcdrop = MCDROP(num_epochs=num_epochs,n_units_1=n_units, n_units_2=n_units, n_units_3=n_units, weight_decay=weight_decay, length_scale=length_scale, T=T, rng=seed)
            model_mcdrop.train(x, y.flatten())
            m_mcdrop, v_mcdrop = model_mcdrop.predict(grid)
            # train with 1 additional new data
            model_mcdrop.train(x_new, y_new.flatten())
            m_mcdrop_new, v_mcdrop_new = model_mcdrop.predict(grid)

            # # -- Train and Prediction with Concrete Dropout Model ---
            model_mcconcdrop = MCCONCRETEDROP(num_epochs=num_epochs, n_units_1=n_units, n_units_2=n_units, n_units_3=n_units,
                                  length_scale=length_scale, T=T, rng=seed)
            model_mcconcdrop.train(x, y.flatten())
            m_mcconcdrop, v_mcconcdrop = model_mcconcdrop.predict(grid)
            # train with 1 additional new data
            model_mcconcdrop.train(x_new, y_new.flatten())
            m_mcconcdrop_new, v_mcconcdrop_new = model_mcconcdrop.predict(grid)


            # -- Train and Prediction with LCBNN with MC Dropout mode and Se_y Util ---
            u1 = 'se_y'
            model_lcbnn_u1 = LCBNN(num_epochs=num_epochs,n_units_1=n_units, n_units_2=n_units, n_units_3=n_units, weight_decay=weight_decay, length_scale=length_scale, T=T, util_type=u1, rng=seed)
            model_lcbnn_u1.train(x, y.flatten())
            m_lcbnn_u1, v_lcbnn_u1 = model_lcbnn_u1.predict(grid)
            # train with 1 additional new data
            model_lcbnn_u1.train(x_new, y_new.flatten())
            m_mcdrop_u1_new, v_mcdrop_u1_new = model_lcbnn_u1.predict(grid)

            # -- Train and Prediction with LCBNN with Se_yclip Util Model ---
            u2 = 'se_yclip'
            model_lcbnn_u2 = LCBNN(num_epochs=num_epochs, n_units_1=n_units, n_units_2=n_units, n_units_3=n_units,
                                   weight_decay=weight_decay, length_scale=length_scale, T=T, util_type=u2, rng=seed)
            model_lcbnn_u2.train(x, y.flatten())
            m_lcbnn_u2, v_lcbnn_u2 = model_lcbnn_u2.predict(grid)
            # train with 1 additional new data
            model_lcbnn_u2.train(x_new, y_new.flatten())
            m_mcdrop_u2_new, v_mcdrop_u2_new = model_lcbnn_u2.predict(grid)

            # -- Plot Results ---
            figure, axes = plt.subplots(2, 3, figsize=(16, 12))


            subplot_titles = ['MC Dropout',
                              'LCBNN '+ u1,
                              'LCBNN '+ u2]

            pred_means_old = [m_mcdrop, m_lcbnn_u1, m_mcconcdrop_new, m_lcbnn_u2]
            pred_var_old = [v_mcdrop, v_lcbnn_u1, m_mcconcdrop_new, v_lcbnn_u2]
            pred_means_new = [m_mcdrop_new, m_mcdrop_u1_new, m_mcconcdrop_new, m_mcdrop_u2_new]
            pred_var_new = [v_mcdrop_new, v_mcdrop_u1_new,m_mcconcdrop_new, v_mcdrop_u2_new]

            pred_means_set = [pred_means_old, pred_means_new]
            pred_var_set = [pred_var_old, pred_var_new]

            # pred_means.append(m_lcbnn_set)
            for j in range(len(pred_means_set)):
                pred_means = pred_means_set[j]
                pred_var  = pred_var_set[j]
                for i in range(len(subplot_titles)):
                    grid = grid.flatten()
                    m = pred_means[i].flatten()
                    v = pred_var[i].flatten()

                    axes[j, i].plot(x, y, "ro")
                    axes[j, i].plot(x_next, y_next, "r^")
                    axes[j, i].plot(grid, fvals, "k--")
                    axes[j, i].plot(grid, pred_means[i], "blue")
                    axes[j, i].fill_between(grid, m + np.sqrt(v), m - np.sqrt(v), color="orange", alpha=0.8)
                    axes[j, i].fill_between(grid, m + 2 * np.sqrt(v), m - 2 * np.sqrt(v), color="orange", alpha=0.6)
                    axes[j, i].fill_between(grid, m + 3 * np.sqrt(v), m - 3 * np.sqrt(v), color="orange", alpha=0.4)
                    axes[j, i].set_title(subplot_titles[i])
                    plt.grid()

            # plt.show()
            path = 'figures_compare/' + func_name + 'n_init='+ str(n_init)+'_l=1e-1_seed' + str(seed) + '_nunits=' + \
                   str(n_units) + '_nepochs=' + str(num_epochs) + '_wd=' + str(weight_decay) + '_dropout=' +str(dropout)
            figure.savefig(path + ".pdf", bbox_inches='tight')

    # plt.savefig(path)
