import math
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize
import matplotlib.pyplot as plt

T = 30
L = 2000000
G = 1.42
C = 3e8
n0 = 1e-9
ne = 1e-8
khaps = 10000
krsui = 2000
p0i = 10
vi0 = 150
ci = np.random.normal(loc=100, scale=10)
c, d = 1000000, 1000000
pui = np.random.rand(50) * 0.1
ai = np.full(50, 100)
bi = np.full(50, 100)
tmaxi = np.random.rand(50) * 30


def compute_constants(xi, yi, lambda_i, i):
    di = np.sqrt((xi - c) ** 2 + (yi - d) ** 2)
    hi = G * (C / (4 * np.pi * di)) ** 2 * 1
    hi_prime = hi * 0.9
    rAi = bi[i] * np.log((n0 * ne + ne * hi * pui[i]) / (n0 * ne + n0 * hi_prime * pui[i]))
    fhapsi = lambda_i * ci / (np.minimum(T - di / vi0, tmaxi[i]) - lambda_i * s / rAi)
    frsui = (1 - lambda_i) * ci / np.minimum(T - di / vi0, tmaxi[i])
    return rAi, fhapsi, frsui


def objective_function(xi, yi, lambda_i, I, s):
    total_energy = 0
    for i in range(I):
        rAi, fhapsi, frsui = compute_constants(xi[i], yi[i], lambda_i[i], i)
        total_energy += (
            p0i * np.sqrt((xi[i] - ai[i]) ** 2 + (yi[i] - bi[i]) ** 2) / vi0 +
            krsui * (frsui ** 2) * (1 - lambda_i[i]) * ci +
            khaps * (fhapsi ** 2) * lambda_i[i] * ci +
            pui[i] * lambda_i[i] * s / rAi
        )
    return total_energy


def optimize_lambda(xi, yi, I, s):
    lambda_i = np.random.rand(I)
    def obj_lambda(lambda_i):
        return objective_function(xi, yi, lambda_i, I, s)
    result = minimize(obj_lambda, lambda_i, bounds=[(0, 1)] * I, method='SLSQP')
    if result.success:
        return result.x
    else:
        raise Exception("Optimization of lambda_i failed")


# s_values = np.linspace(5e4,55e4, 6)
s_values = np.linspace(10,110, 6)
# s_values = np.linspace(50e6,150e6, 6)
car_numbers = 20
xi = np.random.rand(car_numbers) * 1000
yi = np.random.rand(car_numbers) * 1000
propose_values = []
local_values = []
haps_values = []
random_values = []
for s in s_values:
    ci = s

    lambda_propose = optimize_lambda(xi, yi, car_numbers, s)
    energy_propose = objective_function(xi, yi, lambda_propose, car_numbers, s)
    propose_values.append(energy_propose)
    lambda_local = np.zeros(car_numbers)
    energy_local = objective_function(xi, yi, lambda_local, car_numbers, s)
    local_values.append(energy_local)
    lambda_haps = np.ones(car_numbers)
    energy_haps = objective_function(xi, yi, lambda_haps, car_numbers, s)
    haps_values.append(energy_haps)
    lambda_random = np.random.uniform(0, 1, car_numbers)
    energy_random = objective_function(xi, yi, lambda_random, car_numbers, s)
    random_values.append(energy_random)

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'figure.dpi': 300,
    'legend.frameon': True,
    'legend.fancybox': False,
})
fig, ax = plt.subplots(figsize=(3.8, 3))
# s_values = s_values / 1e3
plt.plot(s_values, propose_values, marker='o', linestyle='-', label='Proposed')
plt.plot(s_values, local_values, marker='s', linestyle='--', label='Local')
plt.plot(s_values, haps_values, marker='^', linestyle='-.', label='HAPS')
plt.plot(s_values, random_values, marker='x', linestyle=':', label='Random')
for line in ax.lines:
    line.set_clip_on(False)
plt.xlabel(r'$s_i$ (Mb)')
ax.set_ylabel(r'$E_{\mathrm{total}}$ (J)')
ax.grid(True, linestyle='--', linewidth=0.5)
ax.tick_params(axis='x', direction='in')
ax.tick_params(axis='y', direction='in')
ax.set_xlim(min(s_values), max(s_values))
ax.set_xticks(s_values)
all_y = (propose_values + local_values +
         haps_values + random_values)
max_y = max(all_y)
tick_interval = 500
ymax = math.ceil(max_y / tick_interval) * tick_interval
ax.set_ylim(0, ymax)
ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
# legend = ax.legend(
#     loc='upper right',
#     bbox_to_anchor=(1, 0.84),
#     frameon=True
# )
# legend = ax.legend(loc='upper left', frameon=True)
legend = ax.legend(
    loc='upper right',
    bbox_to_anchor=(0.4, 1),
    frameon=True
)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(1.0)
legend.get_frame().set_linewidth(0.8)
plt.tight_layout(pad=0.3)
plt.savefig('./new_perf_figs/taskSize1.png', dpi=300, bbox_inches='tight')
# plt.show()