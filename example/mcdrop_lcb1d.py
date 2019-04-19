import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from experiments.math_functions import get_function
from pybnn import MCDROP, DNGO, Bohamiann, MCCONCRETEDROP, LCBNN
import time

func_name='gramcy1D'
def f(x_0):
    x = 2*x_0 + 0.5
    return (np.sin(x * 4 * np.pi) / (2*x) + (x-1)**4)-4


total_seeds = 30
length_scale = 1e-1
T = 100
dropout = 0.05
weight_decay = 1e-6
num_epochs = 500
n_units = 100

for s in range(total_seeds):
    print(f'seed={s}')
    n_init = 30
    seed = 42
    rng = np.random.RandomState(s)
    x = rng.rand(n_init)
    y = f(x)
    grid = np.linspace(0, 1, 100)
    fvals = f(grid)
    x = x[:,None]
    grid = grid[:,None]

    start_time = time.time()
    # -- Train and Prediction with MC Model ---
    model_mcdrop = MCDROP(num_epochs=num_epochs,n_units_1=n_units, n_units_2=n_units, n_units_3=n_units,
                          dropout_p=dropout, length_scale = length_scale, T= T,
                          rng=seed, weight_decay=weight_decay)
    model_mcdrop.train(x, y.flatten())
    m_mcdrop, v_mcdrop = model_mcdrop.predict(grid)
    mcdrop_complete_time = time.time()

    # # -- Train and Prediction with MC Model ---
    # model_mcconcdrop = MCCONCRETEDROP(T=T)
    # model_mcconcdrop.train(x, y.flatten())
    # m_mcconcdrop, v_mcconcdrop = model_mcconcdrop.predict(grid)
    # mcconcdrop_complete_time = time.time()

    # # -- Train and Prediction with DNGO Model ---
    # model_dngo = DNGO(do_mcmc=False,rng=seed)
    # model_dngo.train(x, y.flatten(),do_optimize=True)
    # m_dngo, v_dngo = model_dngo.predict(grid)
    # dngo_complete_time = time.time()

    # -- Train and Prediction with LCBNN Model ---
    # -- Train and Prediction with LCBNN None Model ---

    util_set = ['se_y','se_yclip']
    m_lcbnn_set = []
    v_lcbnn_set = []
    loss_lcbnn_set = []
    for u in util_set:
        model_lcbnn = LCBNN(num_epochs=num_epochs,n_units_1=n_units, n_units_2=n_units, n_units_3=n_units,
                            dropout_p=dropout, length_scale = length_scale, T= T,
                            util_type=u ,rng=seed,weight_decay=weight_decay)
        model_lcbnn.train(x, y.flatten())
        m_lcbnn, v_lcbnn = model_lcbnn.predict(grid)
        m_lcbnn_set.append(m_lcbnn.flatten())
        v_lcbnn_set.append(v_lcbnn.flatten())
    # lcbnn_complete_time = time.time()

    # # -- Train and Prediction with Bohamian Model ---
    # model_boham = Bohamiann(print_every_n_steps=1000, rng=seed)
    # model_boham.train(x, y.flatten(), num_steps=4000, num_burn_in_steps=2000, keep_every=50, lr=1e-2, verbose=True)
    # m_boham, v_boham = model_boham.predict(grid)
    # boham_complete_time = time.time()
    #
    # # -- Print Results ---
    # indices = np.where(fvals> 0)
    # loss_mcdrop = np.sqrt(np.mean((m_mcdrop[indices] - fvals[indices])**2))
    # # loss_mcconcdrop = np.sqrt(np.mean((m_mcconcdrop[indices] - fvals[indices])**2))
    # loss_lcbnn = np.sqrt(np.mean((m_lcbnn[indices] - fvals[indices])**2))
    # # loss_boham = np.sqrt(np.mean((m_boham - fvals)**2))

    # print(f'MC_Dropout: time={mcdrop_complete_time-start_time}, rms_loss={loss_mcdrop}')
    # print(f'MC_ConcDropout: time={mcconcdrop_complete_time-mcdrop_complete_time}, rms_loss={loss_mcconcdrop}')
    # print(f'LCBNN: time={lcbnn_complete_time-mcconcdrop_complete_time}, rms_loss={loss_lcbnn}')
    # print(f'Bohamian: time={boham_complete_time-dngo_complete_time}, rms_loss={loss_boham}')

    # -- Plot Results ---
    m_dngo = np.ones_like(m_mcdrop)
    m_boham = np.ones_like(m_mcdrop)
    v_dngo = np.zeros_like(m_mcdrop)
    v_boham = np.zeros_like(m_mcdrop)
    figure, axes = plt.subplots(2, 3, figsize=(16, 12))
    subplot_titles = ['True','MC Dropout','DNGO(BLR)', 'BOHAM(HMC)',
                          'LCBNN '+ util_set[0],
                          'LCBNN '+ util_set[1]]
    pred_means = [fvals, m_mcdrop, m_dngo, m_boham] + m_lcbnn_set
    pred_var = [np.zeros_like(v_mcdrop), v_mcdrop, v_dngo, v_boham] + v_lcbnn_set
    axes_set = [axes[0,0], axes[0,1],axes[0,2], axes[1,0], axes[1,1], axes[1,2]]

    for i in range(len(subplot_titles)):

        grid = grid.flatten()
        m = pred_means[i].flatten()
        v = pred_var[i].flatten()
        axes_set[i].plot(x, y, "ro")
        axes_set[i].plot(grid, fvals, "k--")
        axes_set[i].plot(grid, pred_means[i], "blue")
        axes_set[i].fill_between(grid, m + np.sqrt(v), m - np.sqrt(v), color="orange", alpha=0.8)
        axes_set[i].fill_between(grid, m + 2 * np.sqrt(v), m - 2 * np.sqrt(v), color="orange", alpha=0.6)
        axes_set[i].fill_between(grid, m + 3 * np.sqrt(v), m - 3 * np.sqrt(v), color="orange", alpha=0.4)
        axes_set[i].set_title(subplot_titles[i])
        plt.grid()

    plt.show()
    # path = 'figures/' + func_name + '_l=1e-1_seed' + str(s) + '_nunits=' + \
    #        str(n_units) + '_nepochs=' + str(num_epochs) + '_wd=' + str(weight_decay)
    # figure.savefig(path + ".pdf", bbox_inches='tight')