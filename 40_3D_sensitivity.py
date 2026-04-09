import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter

def dX_dt(vars, t, X_max, C_con, lamda_con, tau_con):
    X, cPDL = vars
    mu_max = C_con - np.exp(lamda_con * cPDL)
    if mu_max >= 0.0:
        dXdt = (1.0 - np.exp(-t / tau_con)) * mu_max * (X_max - X) / X_max * X
        dcPDLdt = (1.0 - np.exp(-t / tau_con)) * mu_max * (X_max - X) / X_max / np.log(2)
    else:
        dXdt = 0.0
        dcPDLdt = 0.0
    return [dXdt, dcPDLdt]

def solve_ode(C_con, lamda_con, tau_con, X0, cPDL0, time_splits):
    X_max = 3.00e5
    initial_conditions = [X0, cPDL0]
    
    X_results = []
    cPDL_results = []
    time_results = []
    
    current_conditions = initial_conditions
    
    for i in range(len(time_splits) - 1):
        time_span = np.linspace(time_splits[i], time_splits[i + 1], num=500)
        solution = odeint(dX_dt, current_conditions, time_span, args=(X_max, C_con, lamda_con, tau_con))
        
        if i == 0:
            X = solution[:, 0]
            cPDL = solution[:, 1]
            X_transit = X[-1]/5
            
        else:
            X = solution[1:, 0]
            cPDL = solution[1:, 1]
            X_transit = X[-1]/5
            time_span = time_span[1:]
        
        current_conditions = [X_transit, cPDL[-1]]
        
        X_results.append(X)
        cPDL_results.append(cPDL)
        time_results.append(time_span)
    
    X_combined = np.concatenate(X_results)
    cPDL_combined = np.concatenate(cPDL_results)
    time_combined = np.concatenate(time_results)
    
    return time_combined, X_combined, cPDL_combined


cell_label = '8331'  # '8580', '2827', '8331'
dir = f'./dataout/{cell_label}'
infile_2D = f'{dir}/fitted_params_CI.csv'
params_2D = pd.read_csv(infile_2D)

V_reactor = 200
X0 = 15000

tau_con_fit = 4.0
C_con_fit = params_2D.loc[params_2D['Parameter'] == 'C', 'Value'].values[0]
lamda_con_fit = params_2D.loc[params_2D['Parameter'] == 'lambda', 'Value'].values[0]
params = C_con_fit, lamda_con_fit, tau_con_fit

if cell_label == '8580':
    p_labels = [3, 4, 5, 6, 7, 8, 9, 10]
elif cell_label == '2827':
    p_labels = [3, 4, 5, 6, 7, 8, 9]
elif cell_label == '8331':
    p_labels = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

t_btb1_array = np.arange(3, 16) # First BtB transfer time [days]

results = []
for p_label in p_labels:
    for t_btb1 in t_btb1_array:
        t_btb2 = t_btb1 + 4 # 2nd BtB transfer [days]
        t_end = 20 # total cultivation days
        time_splits = [0, t_btb1, t_btb2, t_end]

        dir_cPDL0 = f'./data/{cell_label}'
        infile_cPDL0 = f'{dir_cPDL0}/{cell_label}_celldata_mumax.csv'
        df = pd.read_csv(infile_cPDL0)
        cPDL0 = df.loc[df['Label'] == f'p{p_label}', 'cPDL'].values[0]

        time_sim, X_sim, cPDL_sim = solve_ode(*params, X0, cPDL0, time_splits)
        X_btb1 = np.interp(t_btb1, time_sim, X_sim)
        X_btb2 = np.interp(t_btb2, time_sim, X_sim)
        X_end = X_sim[-1]
        n_cells = ((X_btb1 + X_btb2) * 0.8 + X_end) * V_reactor # Total number of cultivated cells

        result = [p_label, t_btb1, t_btb2, t_end, X_btb1, X_btb2, X_end, n_cells]
        results.append(result)

result_df = pd.DataFrame(results, columns=['Passage', 't_btb1_days', 't_btb2_days', 't_end_days',
                                            'X_btb1_cells_per_mL', 'X_btb2_cells_per_mL', 'X_end_cells_per_mL',
                                            'Total_n_cells'])
outdir = './dataout/sensitivity'
os.makedirs(outdir, exist_ok=True)
outfile_csv = f'{outdir}/sensitivity_tbtb1_p_{cell_label}.csv'
result_df.to_csv(outfile_csv, index=False)

x = result_df['Passage'].to_numpy()
y = result_df['t_btb1_days'].to_numpy()
y2 = result_df['t_btb2_days'].to_numpy()
y3 = result_df['t_end_days'].to_numpy()
z = result_df['Total_n_cells'].to_numpy()
z1 = result_df['X_btb1_cells_per_mL'].to_numpy()
z2 = result_df['X_btb2_cells_per_mL'].to_numpy()
z3 = result_df['X_end_cells_per_mL'].to_numpy()

threshold_cells = 0.8e8
binary_z = (z > threshold_cells).astype(int)

Xmax = 3e5
threshold = Xmax * 0.8
binary_z1 = (z1 < threshold).astype(int)
binary_z2 = (z2 < threshold).astype(int)
binary_z3 = (z3 < threshold).astype(int)
binary_all = (binary_z * binary_z1 * binary_z2 * binary_z3)

outfile_csv = f'{outdir}/sensitivity_ds_08e8_{cell_label}.csv'
binary_df = pd.DataFrame({
    'Passage': x,
    't_btb1_days': y,
    't_btb2_days': y2,
    't_end_days': y3,
    'Total_n_cells': z,
    'X_btb1_cells_per_mL': z1,
    'X_btb2_cells_per_mL': z2,
    'X_end_cells_per_mL': z3,
    'Binary_ds': binary_all,
    'Binary_Total_n_cells': binary_z,
    'Binary_X_btb1_cells_per_mL': binary_z1,
    'Binary_X_btb2_cells_per_mL': binary_z2,
    'Binary_X_end_cells_per_mL': binary_z3
})
binary_df.to_csv(outfile_csv, index=False)