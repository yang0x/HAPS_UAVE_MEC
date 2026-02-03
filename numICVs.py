import math

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

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
s = np.random.normal(loc=100, scale=0.1)
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


def objective_function(xi, yi, lambda_i, I):
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


def optimize_lambda(xi, yi, I):
    lambda_i = np.random.rand(I)
    def obj_lambda(lambda_i):
        return objective_function(xi, yi, lambda_i, I)
    result = minimize(obj_lambda, lambda_i, bounds=[(0, 1)] * I, method='SLSQP')
    if result.success:
        return result.x
    else:
        raise Exception("Optimization of lambda_i failed")


def optimize_positions(lambda_i, xi, yi, I):
    def obj_pos(vars):
        xi, yi = vars[:I], vars[I:2 * I]
        return objective_function(xi, yi, lambda_i, I)
    initial_pos = np.concatenate([xi, yi])
    constraints = [
        {'type': 'ineq', 'fun': lambda vars: L ** 2 - ((vars[:I] - c) ** 2 + (vars[I:2 * I] - d) ** 2)},
        {'type': 'ineq', 'fun': lambda vars: np.concatenate(
            [(np.sqrt((vars[:I] - ai[:I]) ** 2 + (vars[I:2 * I] - bi[:I]) ** 2) / vi0 - 0),
             (T - np.sqrt((vars[:I] - ai[:I]) ** 2 + (vars[I:2 * I] - bi[:I]) ** 2) / vi0)])}
    ]
    result = minimize(obj_pos, initial_pos, method='SLSQP', constraints=constraints, options={'maxiter': 1000})
    if result.success:
        return result.x[:I], result.x[I:2 * I]
    else:
        raise Exception("Optimization of positions failed")


def random_lambda_optimization(I):
    return np.random.uniform(0, 1, I)


def print_optimization_details(xi, yi, lambda_i, final_value, I):
    print(f"\nFor {I} cars:")
    print(f"Optimized xi: {xi}")
    print(f"Optimized yi: {yi}")
    print(f"Optimized lambda_i: {lambda_i}")
    print(f"Final objective function value: {final_value}")


car_numbers = np.arange(20, 32,2)
objective_values, local_values, haps_values, random_values = [], [], [], []
xi = np.random.rand(20) * 1000
yi = np.random.rand(20) * 1000
for I in car_numbers:
    if I > 20:
        xi = np.concatenate([xi, np.random.rand(I - len(xi)) * 200])
        yi = np.concatenate([yi, np.random.rand(I - len(yi)) * 200])

    lambda_i = optimize_lambda(xi, yi, I)
    xi_propose, yi_propose = optimize_positions(lambda_i, xi, yi, I)
    final_value = objective_function(xi_propose, yi_propose, lambda_i, I)
    objective_values.append(final_value)

    lambda_local = np.zeros(I)
    xi_local, yi_local = optimize_positions(lambda_local, xi, yi, I)
    local_value = objective_function(xi_local, yi_local, lambda_local, I)
    local_values.append(local_value)

    lambda_haps = np.ones(I)
    xi_haps, yi_haps = optimize_positions(lambda_haps, xi, yi, I)
    haps_value = objective_function(xi_haps, yi_haps, lambda_haps, I)
    haps_values.append(haps_value)

    lambda_random = random_lambda_optimization(I)
    xi_random, yi_random = optimize_positions(lambda_random, xi, yi, I)
    random_value = objective_function(xi_random, yi_random, lambda_random, I)
    random_values.append(random_value)

    # lambda_NonUAVEve = random_lambda_optimization(I)
    # xi_random, yi_random = optimize_positions(lambda_random, xi, yi, I)
    # random_value = objective_function(xi_random, yi_random, lambda_random, I)
    # random_values.append(random_value)

    print_optimization_details(xi_propose, yi_propose, lambda_i, final_value, I)


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
ax.plot(car_numbers, objective_values, marker='o', linestyle='-', label='Proposed')
ax.plot(car_numbers, local_values, marker='s', linestyle='--', label='Local')
ax.plot(car_numbers, haps_values, marker='^', linestyle='-.', label='HAPS')
ax.plot(car_numbers, random_values, marker='x', linestyle=':', label='Random')
for line in ax.lines:
    line.set_clip_on(False)
ax.set_xlabel('Number of ICVs')
ax.set_ylabel(r'$E_{\mathrm{total}}$ ($\mathrm{J}$)')
ax.grid(True, linestyle='--', linewidth=0.5)
ax.tick_params(axis='x', direction='in')
ax.tick_params(axis='y', direction='in')
ax.set_xlim(min(car_numbers), max(car_numbers))
all_y = objective_values + local_values + haps_values + random_values
max_y = max(all_y)
tick_interval = 500
ymax = math.ceil(max_y / tick_interval) * tick_interval
ax.set_ylim(0, ymax)
ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
legend = ax.legend(
    loc='upper right',
    bbox_to_anchor=(1.0, 0.78),
    frameon=True
)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(1.0)
legend.get_frame().set_linewidth(0.8)
plt.tight_layout(pad=0.3)
plt.savefig('./test_figs/numICVs.png', dpi=300, bbox_inches='tight')
# plt.show()