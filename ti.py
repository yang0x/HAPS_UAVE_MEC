import math
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize
import matplotlib.pyplot as plt


T = 100
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


tmaxi_values = np.linspace(10, 40, 7)  # tmaxi值从 10 到 50
objective_values_by_tmaxi = {'Proposed': [], 'Local': [], 'HAPS': [], 'Random': []}
for tmaxi_value in tmaxi_values:
    tmaxi = np.full(50, tmaxi_value)
    xi = np.random.rand(20) * 1000
    yi = np.random.rand(20) * 1000
    lambda_i = optimize_lambda(xi, yi, 20)
    xi_propose, yi_propose = optimize_positions(lambda_i, xi, yi, 20)
    objective_values_by_tmaxi['Proposed'].append(objective_function(xi_propose, yi_propose, lambda_i, 20))
    lambda_local = np.zeros(20)
    xi_local, yi_local = optimize_positions(lambda_local, xi, yi, 20)
    objective_values_by_tmaxi['Local'].append(objective_function(xi_local, yi_local, lambda_local, 20))
    lambda_haps = np.ones(20)
    xi_haps, yi_haps = optimize_positions(lambda_haps, xi, yi, 20)
    objective_values_by_tmaxi['HAPS'].append(objective_function(xi_haps, yi_haps, lambda_haps, 20))
    lambda_random = random_lambda_optimization(20)
    xi_random, yi_random = optimize_positions(lambda_random, xi, yi, 20)
    objective_values_by_tmaxi['Random'].append(objective_function(xi_random, yi_random, lambda_random, 20))

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
ax.plot(tmaxi_values, objective_values_by_tmaxi['Proposed'], marker='o', linestyle='-', label='Proposed')
ax.plot(tmaxi_values, objective_values_by_tmaxi['Local'], marker='s', linestyle='--', label='Local')
ax.plot(tmaxi_values, objective_values_by_tmaxi['HAPS'], marker='^', linestyle='-.', label='HAPS')
ax.plot(tmaxi_values, objective_values_by_tmaxi['Random'], marker='x', linestyle=':', label='Random')
for line in ax.lines:
    line.set_clip_on(False)
plt.xlabel(r'$t_{\mathrm{max,i}}$ ($\mathrm{s}$)')
ax.set_ylabel(r'$E_{\mathrm{total}}$ ($\mathrm{J}$)')
ax.grid(True, linestyle='--', linewidth=0.5)
ax.tick_params(axis='x', direction='in')
ax.tick_params(axis='y', direction='in')
ax.set_xlim(min(tmaxi_values), max(tmaxi_values))
ax.set_xticks(tmaxi_values)
all_y = (objective_values_by_tmaxi['Proposed'] + objective_values_by_tmaxi['Local'] +
         objective_values_by_tmaxi['HAPS'] + objective_values_by_tmaxi['Random'])
max_y = max(all_y)
tick_interval = 400
ymax = math.ceil(max_y / tick_interval) * tick_interval
ax.set_ylim(0, ymax)
ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
legend = ax.legend(
    loc='upper right',
    bbox_to_anchor=(1.0, 0.8),
    frameon=True
)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(1.0)
legend.get_frame().set_linewidth(0.8)
plt.tight_layout(pad=0.3)
plt.savefig('./new_perf_figs/ti.png', dpi=300, bbox_inches='tight')
plt.show()