import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from experiments.math_functions import get_function
from pybnn import DNGO
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization

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

# -- Train Model ---
model = DNGO(do_mcmc=False)
model.train(x, y.flatten(), do_optimize=True)

# -- Predict with Model ---
m, v = model.predict(grid)

figure, axes = plt.subplots(2, 1, figsize=(6, 18))
sub1 = axes[0].contourf(x1, x2, fvals.reshape(50, 50))
axes[0].plot(x[:,0], x[:,1],'rx')
axes[0].set_title('objective func ')

sub2 = axes[1].contourf(x1, x2, m.reshape(50, 50))
axes[1].plot(x[:,0], x[:,1],'rx')
gp_title=f'prediction by DNGO'
axes[1].set_title(gp_title)

plt.show()

# plt.plot(x, y, "ro")
# plt.grid()
# plt.plot(grid, fvals, "k--")
# plt.plot(grid, m, "blue")
# plt.fill_between(grid, m + np.sqrt(v), m - np.sqrt(v), color="orange", alpha=0.8)
# plt.fill_between(grid, m + 2 * np.sqrt(v), m - 2 * np.sqrt(v), color="orange", alpha=0.6)
# plt.fill_between(grid, m + 3 * np.sqrt(v), m - 3 * np.sqrt(v), color="orange", alpha=0.4)
# plt.xlim(0, 1)
# plt.xlabel(r"Input $x$")
# plt.ylabel(r"Output $f(x)$")
# plt.show()