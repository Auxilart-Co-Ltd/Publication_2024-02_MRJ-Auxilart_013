import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter

dir = './dataout/8580'
infile_2D = f'{dir}/fitted_params_CI.csv'
params_2D = pd.read_csv(infile_2D)

for n_dataset in range(5):
    n_dataset += 1
    
    outdir_csv = f'./dataout/prediction'
    outdir_csv2 = f'./dataout/prediction/results'
    outdir_fig = f'./fig/prediction'
    os.makedirs(outdir_csv2, exist_ok=True)
    os.makedirs(outdir_fig, exist_ok=True)
    outfile_csv1 = f'{outdir_csv}/predicted_params_{n_dataset}.csv'
    outfile_csv2 = f'{outdir_csv2}/predicted_results_{n_dataset}.csv'
    outfile_fig = f'{outdir_fig}/cellgrowth_{n_dataset}.png'

    data = pd.read_csv(f'./data/8580_3D/data{n_dataset}.csv')
    time = data.iloc[:, 0].values
    X_data = data.iloc[:, 1].values
    if n_dataset == 1:
        time_splits = [0, 7, 10, time[-1]]
        cPDL0 = 2.03
    elif n_dataset == 2:
        time_splits = [0, 6, 10, time[-1]]
        cPDL0 = 6.06
    elif n_dataset == 3:
        time_splits = [0, 8, 12, time[-1]]
        cPDL0 = 8.43
    elif n_dataset == 4:
        time_splits = [0, 6, 10, time[-1]]
        cPDL0 = 1.90
    elif n_dataset == 5:
        time_splits = [0, 7, 11, time[-1]]
        cPDL0 = 6.46

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

    def solve_ode(C_con, lamda_con, tau_con):
        X_max = 3.00e5
        X0 = X_data[0]
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

    def NRMSE(params):
        C_con, lamda_con, tau_con = params
        time_sim, simulated_X, _ = solve_ode(C_con, lamda_con, tau_con)

        interpolated_X = np.interp(time, time_sim, simulated_X)

        nrmse = np.sqrt(np.mean((X_data - interpolated_X) ** 2)) / (X_data.max() - X_data.min())
        return nrmse

    C_con_fit = params_2D.loc[params_2D['Parameter'] == 'C', 'Value'].values[0]
    lamda_con_fit = params_2D.loc[params_2D['Parameter'] == 'lambda', 'Value'].values[0]
    tau_con_fit = 4.0

    params = C_con_fit, lamda_con_fit, tau_con_fit

    time_sim, X_sim, cPDL_sim = solve_ode(*params)
    nrmse = NRMSE(params)

    data = {
    "Parameter": ["C_con_fit", "lamda_con_fit", "tau_con_fit", "NRMSE"],
    "Value": [C_con_fit, lamda_con_fit, tau_con_fit, nrmse]
    }
    df = pd.DataFrame(data)
    df.to_csv(outfile_csv1, index=False)

    font = {'family': 'Arial', 'size': 18}
    legend_font = {'family': 'Arial', 'size': 18}
    plt.figure(figsize=(9,6))
    plt.plot(time_sim, X_sim, '--', label='Simulation', linewidth=2, color='blue')
    plt.plot(time, X_data, 'ro', label='Experimental data')
    plt.xlabel('Time [days]', fontdict=font)
    plt.ylabel('Cell count [–]', fontdict=font)
    plt.xticks(fontsize=16, fontname='Arial')
    plt.yticks(fontsize=16, fontname='Arial')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.ylim([0,3.0e5])
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((4, 4))  # Force 10^4 formatting
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.offsetText.set_fontsize(16)
    plt.tick_params(left=False, bottom=False)
    # plt.legend(prop=legend_font, frameon=False)
    plt.savefig(outfile_fig, dpi=400, bbox_inches='tight')
    # plt.show()

    result_df = pd.DataFrame({
        'Time': time_sim,
        'X': X_sim,
        'cPDL': cPDL_sim,
    })
    result_df.to_csv(outfile_csv2, index=False)