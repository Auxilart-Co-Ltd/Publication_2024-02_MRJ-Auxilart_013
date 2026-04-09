import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import t
import matplotlib.pyplot as plt
import io
from matplotlib.ticker import ScalarFormatter
import os

i_cell = 2
cell_type = ['2827', '8580', '8331']

# Constants and Initial conditions (keep these constant)
alpha = 0.552
tau = 4.0
beta = -25.2
gamma = 274.0
X_A_seed_base = 1.0
X_A_max = 20.0 * 1.0e+4
V = 100.0
S = 100.0
r_V = 0.99
X_A_seed = 5000.0
X_A_seed_count = 1.35 * 1.0e+6
X_A_ini = X_A_seed
X_S_ini = (X_A_seed_count * r_V - X_A_ini * S) / V

outdir = f'./dataout/{cell_type[i_cell]}'
os.makedirs(f'{outdir}/results', exist_ok=True)
out_file = f'{outdir}/fitted_params.csv'
out_file2 = f'{outdir}/fitted_params_CI.csv'

outdir_figs = f'./fig/{cell_type[i_cell]}'
os.makedirs(outdir_figs, exist_ok=True)
outfile_fig1_1 = f'{outdir_figs}/cell_growth_CI.png'
outfile_fig2 = f'{outdir_figs}/mumax.png'

if cell_type[i_cell] == '2827':
    training_points = 7
    initial_guess = [1.04395641, 0.00201227]
    labels = ['p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']
    labels_fitted = labels[:training_points]
    labels_extrapolated = labels[training_points:]
elif cell_type[i_cell] == '8580':
    training_points = 8
    initial_guess = [1.5, 1.0e-2]
    labels = ['p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
    labels_fitted = labels[:training_points]
    labels_extrapolated = labels[training_points:]
elif cell_type[i_cell] == '8331':
    training_points = 11
    initial_guess = [1.05140876e+00, 6.96995838e-02]
    labels = ['p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13']
    labels_fitted = labels[:training_points]
    labels_extrapolated = labels[training_points:]
else:
    print(f'ERROR: Passage numbers for cell type {cell_type[i_cell]} not defined!')

cell_data = {}
for label in labels:
    infile = f'./data/{cell_type[i_cell]}/{cell_type[i_cell]}_celldata_{label}.csv'
    cell_data[label] = pd.read_csv(infile)

def dX_dt(t, y, t_lag, C_con, lambda_con):
    X_S, X_A, cPDL = y
    Pr_a_val = alpha / tau * np.exp(-t / tau)
    sigma_mu_val = 1 if t >= t_lag else 0
    mu_max = (C_con - np.exp(lambda_con * cPDL)) / 24
    if mu_max >= 0.0:
        mu_val = mu_max * (X_A_max - X_A) / X_A_max
        dX_S_dt = 1.0 / V * (-Pr_a_val * V * X_S)
        dX_A_dt = 1.0 / S * (Pr_a_val * V * X_S + sigma_mu_val * mu_val * S * X_A)
        dcPDL_dt = (sigma_mu_val * mu_val * S * X_A) / ((V * X_S + S * X_A) * np.log(2))
    else:
        dX_S_dt = 0
        dX_A_dt = 0
        dcPDL_dt = 0
    return [dX_S_dt, dX_A_dt, dcPDL_dt]

def solve_ode(C_con, lambda_con, cell_data):
    X_A_seed_count = cell_data[cell_data['time']==0]['cell_num'].values[0]
    X_A_ini = 0.0
    X_S_ini = (X_A_seed_count * r_V) / V
    X_A_seed = X_A_seed_count / S

    t_lag = beta * np.log(X_A_seed / X_A_seed_base) + gamma

    t_span = (0, cell_data['time'].max())
    t_eval = np.linspace(0, t_span[1], 1000)
    cPDL0 = cell_data['cPDL'][0]

    sol = solve_ivp(dX_dt, t_span, [X_S_ini, X_A_ini, cPDL0], args=(t_lag, C_con, lambda_con), t_eval=t_eval)
    cell_number_total_simulated = sol.y[0] * V + sol.y[1] * S
    cPDL_simulated = sol.y[2]

    return t_eval, cell_number_total_simulated, cPDL_simulated

def NRMSE_each(params, cell_data):
    C_con, lambda_con = params
    time_sim, simulated_X, _ = solve_ode(C_con, lambda_con, cell_data)
    time = cell_data['time']
    X_data = cell_data['cell_num']

    interpolated_X = np.interp(time, time_sim, simulated_X)
    mse = (X_data.tail(1) - interpolated_X[-1]) ** 2
    nrmse = np.sqrt(mse) / X_data.max()
    return nrmse

def NRMSE_fitted(params, cell_data_whole):
    nrmse_array = []
    
    for label in labels_fitted:
        nrmse = NRMSE_each(params, cell_data_whole[label])
        nrmse_array.append(nrmse)
                
    nrmse_array = np.array(nrmse_array)
    nrmse = np.mean(nrmse_array)
    return nrmse

bnd = ((0.01, 10), (0.0, 3.0))
result = minimize(NRMSE_fitted, initial_guess, args=(cell_data,), method='Nelder-Mead', bounds=bnd)
print(result)

C_con_fit = result.x[0]
lambda_con_fit = result.x[1]
nrmse_fitted = NRMSE_fitted(result.x, cell_data)

n = len(labels)
p = len(result.x)

residuals = []
for i, label in enumerate(labels):
    time = cell_data[label]['time']
    X_data = cell_data[label]['cell_num']
    time_sim, simulated_X_fit, cPDL_fit = solve_ode(C_con_fit, lambda_con_fit, cell_data[label])
    residual = X_data.iloc[-1] - simulated_X_fit[-1]
    residuals.append(residual)

residuals_np = np.array(residuals)
s_err = np.sqrt(np.sum(residuals_np**2) / (n-p))

J = np.zeros((n, p))
eps = np.sqrt(np.finfo(float).eps)
for i, label in enumerate(labels):
    for j in range(p):
        dp = np.zeros(p)
        dp[j] = eps
        params_dp_pos = result.x + dp
        params_dp_neg = result.x - dp
        time_pos, simulated_X_dp_pos, cPDL_pos = solve_ode(*params_dp_pos, cell_data[label])
        time_neg, simulated_X_dp_neg, cPDL_neg = solve_ode(*params_dp_neg, cell_data[label])
        J[i, j] = (simulated_X_dp_pos[-1] - simulated_X_dp_neg[-1]) / (2 * eps)

a = 0.05
tval = t.ppf(1.0 - a/2., n - p)

JtJ_inv = np.linalg.inv(J.T @ J)
n_ode = 1000

# Confidence interval of the paramaters
pcov = s_err ** 2 * JtJ_inv
params = result.x
perr = np.sqrt(np.diag(pcov))
param_ci = np.array([params - tval * perr, params + tval * perr])
for i, name in enumerate(['C', 'lambda']):
    print(f"  {name}: {param_ci[0, i]:.4f}  –  {param_ci[1, i]:.4f}")

# Creating a DataFrame for the parameters
data = {
    "Parameter": ["C_con_fit", "lamda_con_fit", "NRMSE_fitted", "C_SE", "C_CI", "C_bnd_min", "C_bnd_max", "lamda_SE", "lamda_CI", "lamda_bnd_min", "lamda_bnd_max"],
    "Value": [C_con_fit, lambda_con_fit, nrmse_fitted, perr[0], tval*perr[0], bnd[0][0], bnd[0][1], perr[1], tval*perr[1], bnd[1][0], bnd[1][1]]
}
df = pd.DataFrame(data)
df.to_csv(out_file, index=False)

# Creating a DataFrame for the parameters
data_ci = {
    "Parameter": ["C_lower", "C", "C_upper", "lambda_lower", "lambda", "lambda_upper"],
    "Value": [param_ci[0, 0], C_con_fit, param_ci[1,0], param_ci[0,1], lambda_con_fit, param_ci[1,1]]
}
df_ci = pd.DataFrame(data_ci)
df_ci.to_csv(out_file2, index=False)

if cell_type[i_cell] == '2827':
    fig, axs = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    axs = axs.flatten()
    ymax = 1.0e7
elif cell_type[i_cell] == '8580':
    fig, axs = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    axs = axs.flatten()
    ymax = 1.0e7
elif cell_type[i_cell] == '8331':
    fig, axs = plt.subplots(3, 4, figsize=(20, 12), sharey=True)
    axs = axs.flatten()
    ymax = 1.8e7

for i, label in enumerate(labels):
    time = cell_data[label]['time']
    X_data = cell_data[label]['cell_num']
    J_sim = np.zeros((n_ode, p))
    ci = []
    pi = []
    for j in range(p):
        dp = np.zeros(p)
        dp[j] = eps
        params_dp_pos = result.x + dp
        params_dp_neg = result.x - dp
        time_pos, simulated_X_dp_pos, cPDL_pos = solve_ode(*params_dp_pos, cell_data[label])
        time_neg, simulated_X_dp_neg, cPDL_neg = solve_ode(*params_dp_neg, cell_data[label])
        for i_x in range(n_ode):
            J_sim[i_x, j] = (simulated_X_dp_pos[i_x] - simulated_X_dp_neg[i_x]) / (2 * eps)
    
    for i_x in range(n_ode):
        g = J_sim[i_x, :] @ JtJ_inv @ J_sim[i_x, :].T
        dy_conf = tval * s_err * np.sqrt(g)
        dy_pred = tval * s_err * np.sqrt(1 + g)
        ci.append(dy_conf)
        pi.append(dy_pred)
    ci = np.array(ci)
    pi = np.array(pi)

    time_sim, simulated_X, cPDL_fit = solve_ode(C_con_fit, lambda_con_fit, cell_data[label])

    axs[i].plot(time, X_data, 'ro', label='Observed')
    axs[i].plot(time_sim, simulated_X, label='Simulated')
    axs[i].fill_between(time_sim, simulated_X - ci, simulated_X + ci,
                        color='blue', alpha=0.1, label='95% Model CI')
    axs[i].set_title(f'Passage = {label}',fontsize=18)
    axs[i].set_xlabel('Time', fontsize=16)
    axs[i].set_ylabel('Cell Count', fontsize=16)
    axs[i].set_ylim(0,ymax)
    #axs[i].legend(fontsize=14)
    axs[i].tick_params(axis='both', which='major', labelsize=14)

    result_df = pd.DataFrame({
    'Time': time_sim,
    'X': simulated_X,
    'cPDL': cPDL_fit,
    '95% CI': ci,
    'X_CI_upper': simulated_X + ci,
    'X_CI_lower': simulated_X - ci,
    '95% PI': pi,
    'X_PI_lower': simulated_X - pi,
    'X_PI_upper': simulated_X + pi
    })
    result_df.to_csv(f'{outdir}/results/fitted_results_{label}.csv', index=False)

for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.savefig(outfile_fig1_1, dpi=400, bbox_inches='tight')
# plt.show()

infile = f'./data/{cell_type[i_cell]}/{cell_type[i_cell]}_celldata_mumax.csv'
mumax_df = pd.read_csv(infile)
cPDL_data = mumax_df['cPDL']
mumax_data = mumax_df['mu_max']
cPDL_data = np.array(cPDL_data)
mumax_data = np.array(mumax_data)
mumax_data = mumax_data * 24 # per day

## Calculate the 95% model CI for mu_max
def f_cPDL_2D(cPDL, conc, lamda):
    mu_max = (conc - np.exp(cPDL * lamda))
    return mu_max # per day

params = result.x
cPDL_pred = np.linspace(0, 40, 100)
mumax_pred = f_cPDL_2D(cPDL_pred, *params)

residuals_mu = mumax_data - f_cPDL_2D(cPDL_data, *params)
s_err_mu = np.sqrt(np.sum(residuals_mu**2) / (n - p))

J_mu = np.zeros((n, p))
eps = np.sqrt(np.finfo(float).eps)
for i in range(p):
    dp = np.zeros(p)
    dp[i] = eps
    J_mu[:, i] = (f_cPDL_2D(cPDL_data, *(params + dp)) - f_cPDL_2D(cPDL_data, *(params - dp))) / (2 * eps)

a = 0.05
tval = t.ppf(1.0 - a/2., n - p)

JtJ_inv = np.linalg.inv(J_mu.T @ J_mu)
ci = []
pi = []
for cPDL0 in cPDL_pred:
    Jx = np.zeros(p)
    for i in range(p):
        dp = np.zeros(p)
        dp[i] = eps
        Jx[i] = (f_cPDL_2D(cPDL0, *(params + dp)) - f_cPDL_2D(cPDL0, *(params - dp))) / (2 * eps)
    g = Jx @ JtJ_inv @ Jx.T
    dy_conf = tval * s_err_mu * np.sqrt(g)
    dy_pred = tval * s_err_mu * np.sqrt(1 + g)
    ci.append(dy_conf)
    pi.append(dy_pred)
ci = np.array(ci)
pi = np.array(pi)

result_mu_df = pd.DataFrame({
    'cPDL': cPDL_pred,
    'mu_max': mumax_pred,
    '95% CI': ci,
    'mu_max_CI_upper': mumax_pred + ci,
    'mu_max_CI_lower': mumax_pred - ci,
    '95% PI': pi,
    'mu_max_PI_lower': mumax_pred - pi,
    'mu_max_PI_upper': mumax_pred + pi
})
result_mu_df.to_csv(f'{outdir}/results/fitted_mumax_CI.csv', index=False)

plt.figure(figsize=(8,8))
plt.plot(cPDL_pred, mumax_pred, 'b')
plt.plot(cPDL_data, mumax_data, 'o', color='red')
plt.fill_between(cPDL_pred, mumax_pred - ci, mumax_pred + ci, color='skyblue', alpha=0.3)
plt.xlabel('cPDL', fontsize=14)
plt.ylabel('mu_max [day$^{-1}$]', fontsize=14)
plt.xlim(0,40)
plt.ylim(0,1.4)
plt.tick_params(axis='both', which='both', labelsize=12, direction='in', size=8)
plt.xticks([0,10,20,30,40], fontsize=14, y=-0.02)
plt.yticks(fontsize=14, x=-0.01)
plt.savefig(outfile_fig2, dpi=400, bbox_inches='tight')
# plt.show()