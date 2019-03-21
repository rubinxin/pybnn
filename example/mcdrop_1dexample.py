import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pybnn import MCDROP
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization

# plt.rc('text', usetex=True)
# plt.rc('font', size=12.0, family='serif')
# plt.rcParams['figure.figsize'] = (8.0, 6.0)
# plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

def f(x):
    return np.sinc(x * 10 - 5)

rng = np.random.RandomState(42)

x = rng.rand(20)
y = f(x)

grid = np.linspace(0, 1, 100)
fvals = f(grid)

plt.plot(grid, fvals, "k--")
plt.plot(x, y, "ro")
plt.grid()
plt.xlim(0, 1)
plt.show()

# -- Train Model ---
model = MCDROP()
model.train(x[:, None], y)

# -- Predict with Model ---
m, v = model.predict(grid[:, None])

plt.plot(x, y, "ro")
plt.grid()
plt.plot(grid, fvals, "k--")
plt.plot(grid, m, "blue")
plt.fill_between(grid, m + np.sqrt(v), m - np.sqrt(v), color="orange", alpha=0.8)
plt.fill_between(grid, m + 2 * np.sqrt(v), m - 2 * np.sqrt(v), color="orange", alpha=0.6)
plt.fill_between(grid, m + 3 * np.sqrt(v), m - 3 * np.sqrt(v), color="orange", alpha=0.4)
plt.xlim(0, 1)
plt.xlabel(r"Input $x$")
plt.ylabel(r"Output $f(x)$")
plt.show()
