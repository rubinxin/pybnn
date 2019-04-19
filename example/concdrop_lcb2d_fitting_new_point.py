import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from experiments.math_functions import get_function
from pybnn import MCDROP, DNGO, Bohamiann, MCCONCRETEDROP,LCBNN, LCCD
import time
from matplotlib import cm

func_name = 'branin-2d'
# func_name = 'egg-2d'

func, bounds, x_true_min, true_fmin = get_function(func_name)
x_true_min = np.atleast_2d(x_true_min)
f = lambda x: (func(x) - 2)* 10
d = bounds.shape[0]
n_init = 60
var_noise = 1.0e-10

total_seeds = 10
length_scale = 1e-1
T = 100
dropout = 0.05
weight_decay = 1e-6
num_epochs_set = [500,1000,5000]
n_units_set = [10,50,100]

for num_epochs in num_epochs_set:
    for n_units in n_units_set:
        for s in range(1,total_seeds):
            print(f'seed={s}')
            seed = s
            np.random.seed(42)
            x = np.random.uniform(bounds[:,0], bounds[:,1], (n_init, d))
            y = f(x)
            x1, x2 = np.mgrid[-1:1:50j, -1:1:50j]
            grid = np.vstack((x1.flatten(), x2.flatten())).T
            fvals = f(grid)


            # Add 1 new point
            x_next = np.atleast_2d(x_true_min[-1])
            x_next -= 0.1*np.random.rand(len(x_next))
            y_next = f(x_next)
            x_new  = np.vstack((x, x_next))
            y_new  = np.vstack((y, y_next))

            plt.contourf(x1, x2, fvals.reshape(50, 50), cmap=cm.PuBu_r)
            plt.plot(x[:, 0], x[:, 1], 'gx')
            plt.plot(x_next[:, 0], x_next[:, 1], 'k^')
            plt.show()

            # -- Train and Prediction with MC Model ---
            model_mcdrop = MCDROP(num_epochs=num_epochs,n_units_1=n_units, n_units_2=n_units, n_units_3=n_units, weight_decay=weight_decay, length_scale=length_scale, T=T, rng=seed)
            model_mcdrop.train(x, y.flatten())
            m_mcdrop, v_mcdrop = model_mcdrop.predict(grid)
            # train with 1 additional new data
            model_mcdrop.train(x_new, y_new.flatten())
            m_mcdrop_new, v_mcdrop_new = model_mcdrop.predict(grid)

            # -- Train and Prediction with LCBNN with Se_y Util Model ---
            u1 = 'se_y'
            model_lcbnn_u1 = LCBNN(num_epochs=num_epochs,n_units_1=n_units, n_units_2=n_units, n_units_3=n_units, weight_decay=weight_decay, length_scale=length_scale, T=T, util_type=u1, rng=seed)
            model_lcbnn_u1.train(x, y.flatten())
            m_lcbnn_u1, v_lcbnn_u1 = model_lcbnn_u1.predict(grid)
            # train with 1 additional new data
            model_lcbnn_u1.train(x_new, y_new.flatten())
            m_mcdrop_u1_new, v_mcdrop_u1_new = model_lcbnn_u1.predict(grid)

            # -- Train and Prediction with Concrete Dropout Model ---
            model_mcconcdrop = MCCONCRETEDROP(num_epochs=num_epochs, n_units_1=n_units, n_units_2=n_units,
                                              n_units_3=n_units, length_scale=length_scale, T=T, rng=seed)
            model_mcconcdrop.train(x, y.flatten())
            m_mcconcdrop, v_mcconcdrop = model_mcconcdrop.predict(grid)
            # train with 1 additional new data
            model_mcconcdrop.train(x_new, y_new.flatten())
            m_mcconcdrop_new, v_mcconcdrop_new = model_mcconcdrop.predict(grid)

            # -- Train and Prediction with LCBNN with Concrete Dropout Model with Se_y Util ---
            model_lccd_u1 = LCCD(num_epochs=num_epochs, n_units_1=n_units, n_units_2=n_units, n_units_3=n_units,
                                 length_scale=length_scale, T=T, util_type=u1, rng=seed)
            model_lccd_u1.train(x, y.flatten())
            m_lccd_u1, v_lccd_u1 = model_lccd_u1.predict(grid)
            # train with 1 additional new data
            model_lccd_u1.train(x_new, y_new.flatten())
            m_mcdrop_u2_new, v_mcdrop_u2_new = model_lccd_u1.predict(grid)

            # -- Plot Results ---
            figure, axes = plt.subplots(2, 4, figsize=(20, 10))

            subplot_titles = ['MC Dropout', 'LCBNN ' + u1, 'Conc. Dropout', 'LCCD' + u1]

            pred_means_old = [m_mcdrop, m_lcbnn_u1, m_mcconcdrop, m_lccd_u1]
            pred_means_new = [m_mcdrop_new, m_mcdrop_u1_new, m_mcconcdrop_new, m_mcdrop_u2_new]
            pred_means_set = [pred_means_old, pred_means_new]
            # pred_means.append(m_lcbnn_set)
            for j in range(len(pred_means_set)):
                pred_means = pred_means_set[j]
                for i in range(len(subplot_titles)):
                    ax_i = axes[j,i].contourf(x1, x2, pred_means[i].reshape(50, 50), cmap=cm.PuBu_r)
                    axes[j,i].plot(x[:,0], x[:,1],'gx')
                    axes[j,i].plot(x_next[:,0], x_next[:,1],'k^')
                    axes[j,i].set_title(subplot_titles[i])
                    axes[j,i].plot(x_true_min[:,0], x_true_min[:,1],'ro', markersize=6, markerfacecoloralt='r')
                    figure.colorbar(ax_i, ax=axes[j,i])

            # plt.show()
            path = 'figures/' + func_name + '_l=1e-1_seed' + str(seed) + '_nunits=' + \
                   str(n_units) + '_nepochs=' + str(num_epochs) + '_wd=' + str(weight_decay)
            figure.savefig(path + ".pdf", bbox_inches='tight')

    # plt.savefig(path)
