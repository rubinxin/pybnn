import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from experiments.math_functions import get_function
from pybnn import MCDROP, DNGO, Bohamiann
import time

f = lambda x: np.sinc(x * 10 - 5)
n_init = 20
rng = np.random.RandomState(42)
x = rng.rand(n_init)
y = f(x)
grid = np.linspace(0, 1, 100)
fvals = f(grid)
x = x[:,None]
grid = grid[:,None]

start_time = time.time()
# -- Train and Prediction with MC Model ---
tau = 1.0
T = 100
dropout = 0.05
model_mcdrop = MCDROP(dropout=dropout, tau = tau, T= T)
model_mcdrop.train(x, y.flatten())
m_mcdrop, v_mcdrop = model_mcdrop.predict(grid)
mcdrop_complete_time = time.time()

# -- Train and Prediction with DNGO Model ---
model_dngo = DNGO(do_mcmc=False)
model_dngo.train(x, y.flatten(),do_optimize=True)
m_dngo, v_dngo = model_dngo.predict(grid)
dngo_complete_time = time.time()

# -- Train and Prediction with Bohamian Model ---
model_boham = Bohamiann(print_every_n_steps=1000)
model_boham.train(x, y.flatten(), num_steps=6000, num_burn_in_steps=2000, keep_every=50, lr=1e-2, verbose=True)
m_boham, v_boham = model_boham.predict(grid)
boham_complete_time = time.time()

# -- Print Results ---
loss_mcdrop = np.sqrt(np.mean((m_mcdrop - fvals)**2))
loss_dngo = np.sqrt(np.mean((m_dngo - fvals)**2))
loss_boham = np.sqrt(np.mean((m_boham - fvals)**2))

print(f'MC_Dropout: time={mcdrop_complete_time-start_time}, rms_loss={loss_mcdrop}')
print(f'DNGO_Dropout: time={dngo_complete_time-mcdrop_complete_time}, rms_loss={loss_dngo}')
print(f'Bohamian_Dropout: time={boham_complete_time-dngo_complete_time}, rms_loss={loss_boham}')

# -- Plot Results ---
figure, axes = plt.subplots(2, 2, figsize=(6, 18))
subplot_titles = ['True Func, RMSE='+str(round(0.00,3)),
                  'MC Dropout, RMSE='+str(round(loss_mcdrop,3)),
                  'DNGO, RMSE='+str(round(loss_dngo,3)),
                  'Bohamiann, RMSE='+str(round(loss_boham,3))]
pred_means = [fvals, m_mcdrop, m_dngo, m_boham]
pred_var = [np.atleast_2d(0), v_mcdrop, v_dngo, v_boham]
axes_set = [axes[0,0], axes[0,1],axes[1,0],axes[1,1]]
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
