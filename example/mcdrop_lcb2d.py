import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from experiments.math_functions import get_function
from pybnn import MCDROP, DNGO, Bohamiann, MCCONCRETEDROP,LCBNN
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

total_seeds = 30
length_scale = 1e-1
T = 200
dropout = 0.05
weight_decay = 1e-6
num_epochs = 500
n_units = 100

for s in range(total_seeds):
    print(f'seed={s}' )
    seed = s
    np.random.seed(42)
    x = np.random.uniform(bounds[:,0], bounds[:,1], (n_init, d))
    y = f(x)
    x1, x2 = np.mgrid[-1:1:50j, -1:1:50j]
    grid = np.vstack((x1.flatten(), x2.flatten())).T
    fvals = f(grid)

    start_time = time.time()
    # -- Train and Prediction with MC Model ---
    model_mcdrop = MCDROP(num_epochs=num_epochs,n_units_1=n_units, n_units_2=n_units, n_units_3=n_units, weight_decay=weight_decay, length_scale=length_scale, T=T, rng=seed)
    model_mcdrop.train(x, y.flatten())
    m_mcdrop, v_mcdrop = model_mcdrop.predict(grid)
    mcdrop_complete_time = time.time()

    # -- Train and Prediction with LCBNN None Model ---
    util_set = ['se_y','se_yclip']
    m_lcbnn_set = []
    loss_lcbnn_set = []
    for u in util_set:
        model_lcbnn= LCBNN(num_epochs=num_epochs,n_units_1=n_units, n_units_2=n_units, n_units_3=n_units, weight_decay=weight_decay, length_scale=length_scale, T=T, util_type=u,rng=seed)
        model_lcbnn.train(x, y.flatten())
        m_lcbnn, v_lcbnn = model_lcbnn.predict(grid)
        lcbnn_complete_time = time.time()
        m_lcbnn_set.append(m_lcbnn.flatten())
        loss_lcbnn = np.sqrt(np.mean((m_lcbnn - fvals.flatten()) ** 2))
        loss_lcbnn_set.append(loss_lcbnn)

    # # -- Train and Prediction with DNGO Model ---
    model_dngo = DNGO(num_epochs=num_epochs,n_units_1=n_units, n_units_2=n_units, n_units_3=n_units, do_mcmc=False,rng=seed)
    model_dngo.train(x, y.flatten(),do_optimize=True)
    m_dngo, v_dngo = model_dngo.predict(grid)
    dngo_complete_time = time.time()
    #
    # # -- Train and Prediction with Bohamian Model ---
    model_boham = Bohamiann(print_every_n_steps=1000, rng=seed)
    model_boham.train(x, y.flatten(), num_steps=6000, num_burn_in_steps=2000, keep_every=50, lr=1e-2, verbose=True)
    m_boham, v_boham = model_boham.predict(grid)
    boham_complete_time = time.time()

    # -- Print Results ---

    loss_mcdrop = np.sqrt(np.mean((m_mcdrop - fvals.flatten())**2))
    loss_lcbnn = np.sqrt(np.mean((m_lcbnn - fvals.flatten())**2))
    # loss_mcconcdrop = np.sqrt(np.mean((m_mcconcdrop - fvals)**2))
    loss_dngo = np.sqrt(np.mean((m_dngo - fvals.flatten())**2))
    loss_boham = np.sqrt(np.mean((m_boham - fvals.flatten())**2))

    # print(f'MC_Dropout: time={mcdrop_complete_time-start_time}, rms_loss={loss_mcdrop}')
    # print(f'MC_ConcDropout: time={mcconcdrop_complete_time-mcdrop_complete_time}, rms_loss={loss_mcconcdrop}')
    # print(f'DNGO: time={dngo_complete_time-mcdrop_complete_time}, rms_loss={loss_dngo}')
    # print(f'Bohamian: time={boham_complete_time-dngo_complete_time}, rms_loss={loss_boham}')

    # -- Plot Results ---
    figure, axes = plt.subplots(2, 3, figsize=(16, 12))
    # subplot_titles = ['True',
    #                   'MC Dropout, RMSE='+str(round(loss_mcdrop,3)),
    #                   # 'MC_ConcDropout, RMSE='+str(round(loss_mcconcdrop,3)),
    #                   'DNGO, RMSE='+str(round(loss_dngo,3)),
    #                   'LCBNN, RMSE=' + str(round(loss_lcbnn, 3)),
    #                   'LCBNN, RMSE=' + str(round(loss_lcbnn, 3)),
    #                   'LCBNN, RMSE=' + str(round(loss_lcbnn, 3)),
    #
    #                   ]
                      # 'Bohamiann, RMSE='+str(round(loss_boham,3))]

    subplot_titles = ['True','MC Dropout','DNGO(BLR)', 'BOHAM(HMC)',
                      'LCBNN '+ util_set[0],
                      'LCBNN '+ util_set[1]]

    pred_means = [fvals, m_mcdrop, m_dngo, m_boham] + m_lcbnn_set
    # pred_means.append(m_lcbnn_set)
    axes_set = [axes[0,0], axes[0,1],axes[0,2], axes[1,0], axes[1,1], axes[1,2]]
    for i in range(len(subplot_titles)):
        ax_i = axes_set[i].contourf(x1, x2, pred_means[i].reshape(50, 50), cmap=cm.PuBu_r)
        axes_set[i].plot(x[:,0], x[:,1],'gx')
        axes_set[i].set_title(subplot_titles[i])
        axes_set[i].plot(x_true_min[:,0], x_true_min[:,1],'ro', markersize=6, markerfacecoloralt='r')
        figure.colorbar(ax_i, ax=axes_set[i])
    # plt.show()
    path = 'figures/' + func_name + '_l=1e-1_seed' + str(seed) + '_nunits=' + \
           str(n_units) + '_nepochs=' + str(num_epochs) + '_wd=' + str(weight_decay)
    figure.savefig(path + ".pdf", bbox_inches='tight')

    # plt.savefig(path)
