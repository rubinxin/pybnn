import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from experiments.math_functions import get_function
from pybnn import MCDROP, DNGO, Bohamiann
import time

func_name = 'camelback-2d'
f, bounds, _, true_fmin = get_function(func_name)
d = bounds.shape[0]
n_init = 40
var_noise = 1.0e-10
np.random.seed(3)
x = np.random.uniform(bounds[:,0], bounds[:,1], (n_init, d))
y = f(x)
x1, x2 = np.mgrid[-1:1:50j, -1:1:50j]
grid = np.vstack((x1.flatten(), x2.flatten())).T
fvals = f(grid)


start_time = time.time()
# -- Train and Prediction with MC Model ---
model_mcdrop = MCDROP()
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
loss_mcdrop = np.sqrt(np.mean((m_mcdrop - fvals.flatten())**2))
loss_dngo = np.sqrt(np.mean((m_dngo - fvals.flatten())**2))
loss_boham = np.sqrt(np.mean((m_boham - fvals.flatten())**2))

print(f'MC_Dropout: time={mcdrop_complete_time-start_time}, rms_loss={loss_mcdrop}')
print(f'DNGO_Dropout: time={dngo_complete_time-mcdrop_complete_time}, rms_loss={loss_dngo}')
print(f'Bohamian_Dropout: time={boham_complete_time-dngo_complete_time}, rms_loss={loss_boham}')

# -- Plot Results ---
figure, axes = plt.subplots(2, 2, figsize=(6, 18))
subplot_titles = ['True Func , RMSE='+str(round(0.000,3)),
                'MC Dropout, RMSE='+str(round(loss_mcdrop,3)),
                'DNGO, RMSE='+str(round(loss_dngo,3)),
                'Bohamiann, RMSE='+str(round(loss_boham,3))]

pred_means = [fvals, m_mcdrop, m_dngo, m_boham]
axes_set = [axes[0,0], axes[0,1],axes[1,0],axes[1,1]]
for i in range(len(subplot_titles)):
    axes_set[i].contourf(x1, x2, pred_means[i].reshape(50, 50))
    axes_set[i].plot(x[:,0], x[:,1],'rx')
    axes_set[i].set_title(subplot_titles[i])

plt.show()
