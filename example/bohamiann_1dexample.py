import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pybnn.bohamiann import Bohamiann

def f(x):
 return np.sinc(x * 10 - 5)

rng = np.random.RandomState(42)

x = rng.rand(20)
y = f(x)

grid = np.linspace(0, 1, 200)
fvals = f(grid)

plt.plot(grid, fvals, "k--")
plt.plot(x, y, "ro")
plt.grid()
plt.xlim(0, 1)

plt.show()


# -- Train Model ---
model = Bohamiann(print_every_n_steps=1000)
model.train(x[:, None], y, num_steps=20000, num_burn_in_steps=2000, keep_every=50, lr=1e-2, verbose=True)

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

# -- Get Prediction Samples --
# m, v, samples = model.predict(grid[:, None], return_individual_predictions=True)
# print(samples.shape)
# for sample in samples:
#     plt.plot(grid, sample, "blue", alpha=0.2)
#
# plt.plot(x, y, "ro")
# plt.grid(True)
# plt.plot(grid, fvals, "k--")
# plt.xlim(0, 1)
# plt.xlabel(r"Input $x$")
# plt.ylabel(r"Output $f(x)$")
# plt.show()