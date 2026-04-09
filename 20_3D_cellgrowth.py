import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
from matplotlib.ticker import ScalarFormatter
import csv
from scipy.stats import t

fitting_conditions = 0

while n_dataset <= 5:
    header1 = f'./dataout/fitting'
    header2 = f'./fig/fitting'
    if fitting_conditions == 0:
        outdir1 = f'{header1}'
        outdir2 = f'{header1}/results'
        outdir_fig = f'{header2}'

    os.makedirs(outdir1, exist_ok=True)
    os.makedirs(outdir2, exist_ok=True)
    os.makedirs(outdir_fig, exist_ok=True)
    csv_file1 = f'{outdir1}/fitted_params_{n_dataset}.csv'
    csv_file2 = f'{outdir1}/fitted_params_CI_{n_dataset}.csv'
    csv_file3 = f'{outdir1}/results/fitted_results_{n_dataset}.csv'
    png_file1 = f'{outdir_fig}/3D_cellgrowth_CI_{n_dataset}.png'

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

    tau = 4.0

    def dX_dt(vars, t, X_max, C_con, lamda_con):
        X, cPDL = vars
        mu_max = C_con - np.exp(lamda_con * cPDL)
        if mu_max >= 0.0:
            dXdt = (1.0 - np.exp(-t / tau)) * mu_max * (X_max - X) / X_max * X
            dcPDLdt = (1.0 - np.exp(-t / tau)) * mu_max * (X_max - X) / X_max / np.log(2)
        else:
            dXdt = 0.0
            dcPDLdt = 0.0
        return [dXdt, dcPDLdt]

    def solve_ode(C_con, lamda_con):
        X_max = 3.00e5
        X0 = X_data[0]
        initial_conditions = [X0, cPDL0]
        
        X_results = []
        cPDL_results = []
        time_results = []
        
        current_conditions = initial_conditions
        
        for i in range(len(time_splits) - 1):
            time_span = np.linspace(time_splits[i], time_splits[i + 1], num=500)
            solution = odeint(dX_dt, current_conditions, time_span, args=(X_max, C_con, lamda_con))
            
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
        C_con, lamda_con = params
        time_sim, simulated_X, _ = solve_ode(C_con, lamda_con)
        interpolated_X = np.interp(time, time_sim, simulated_X)
        nrmse = np.sqrt(np.mean((X_data - interpolated_X) ** 2)) / (X_data.max() - X_data.min())
        return nrmse

    bnds = ((1.60, 1.70), (1.0e-2, 1.4e-2))
    initial_guess = [1.6127576008923308, 0.01]
    if fitting_conditions == 0:
        result = minimize(NRMSE, initial_guess, method='Nelder-Mead', bounds = bnds)

    C_con_fit, lamda_con_fit = result.x
    nrmse = NRMSE(result.x)

    def numerical_jacobian(model_func, params, time_points, eps=np.sqrt(np.finfo(float).eps)):
        p = len(params)
        J = np.zeros((len(time_points), p))
        for i in range(p):
            dp = np.zeros(p)
            dp[i] = eps
            t_pos, X_pos, _ = model_func(* (params + dp))
            t_neg, X_neg, _ = model_func(* (params - dp))
            X_pos_inter = np.interp(time_points, t_pos, X_pos)
            X_neg_inter = np.interp(time_points, t_neg, X_neg)
            J[:, i] = (X_pos_inter - X_neg_inter) / (2 * eps)
        return J

    def calc_conf_pred_interval(model_func, params, time_data, X_data, a=0.05):
        n = len(X_data)
        p = len(params)

        t_sim, X_sim, _ = model_func(*params)
        X_interp = np.interp(time_data, t_sim, X_sim)
        residuals = X_data - X_interp
        s_err = np.sqrt(np.sum(residuals**2) / (n - p))

        J = numerical_jacobian(model_func, params, time_data)
        JtJ_inv = np.linalg.inv(J.T @ J)

        tval = t.ppf(1.0 - a/2., n - p)

        ci, pi = [], []
        for t0 in t_sim:
            Jx = numerical_jacobian(model_func, params, np.array([t0]))[0]
            g = Jx @ JtJ_inv @ Jx.T
            dy_conf = tval * s_err * np.sqrt(g)
            dy_pred = tval * s_err * np.sqrt(1 + g)
            ci.append(dy_conf)
            pi.append(dy_pred)

        pcov = s_err**2 * JtJ_inv
        perr = np.sqrt(np.diag(pcov))
        params_ci = ([params - tval * perr, params + tval * perr])

        return np.array(ci), np.array(pi), params_ci, perr, tval

    params = result.x
    time_sim, X_sim, cPDL_sim = solve_ode(*params)
    ci, pi, params_ci, perr, tval = calc_conf_pred_interval(solve_ode, params, time, X_data)

    data = {
        "Parameter": ["C_con_fit", "lamda_con_fit", "tau_con_fit", "NRMSE", "C_bnd_min", "C_bnd_max", "lamda_bnd_min", "lamda_bnd_max"],
        "Value": [C_con_fit, lamda_con_fit, tau, nrmse, bnds[0][0], bnds[0][1], bnds[1][0], bnds[1][1]]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file1, index=False)

    data_ci = {
        "Parameter": ["C_SE", "C_CI", "C_lower", "C", "C_upper", "lambda_SE", "lambda_CI", "lambda_lower", "lambda", "lambda_upper"],
        "Value": [perr[0], tval*perr[0], params_ci[0][0], C_con_fit, params_ci[1][0], perr[1], tval*perr[1], params_ci[0][1], lamda_con_fit, params_ci[1][1]]
    }
    df_ci = pd.DataFrame(data_ci)
    df_ci.to_csv(csv_file2, index=False)

    font = {'family': 'Arial', 'size': 18}
    legend_font = {'family': 'Arial', 'size': 18}
    plt.figure(figsize=(8, 6))
    plt.plot(time, X_data, 'ro', label='Experimental data')
    plt.plot(time_sim, X_sim, '-', label='Simulation', linewidth=2, color='blue')
    plt.fill_between(time_sim, X_sim - ci, X_sim + ci, 
                    color='blue', alpha=0.1, label='95% CI')
    plt.xlabel('Time [days]', fontdict=font)
    plt.ylabel('Cell count [–]', fontdict=font)
    plt.xticks(fontsize=16, fontname='Arial')
    plt.yticks(fontsize=16, fontname='Arial')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((4, 4))  # Force 10^4 formatting
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.offsetText.set_fontsize(16)
    plt.tick_params(left=False, bottom=False)
    plt.legend(prop=legend_font, frameon=False)
    plt.savefig(png_file1, dpi=400, bbox_inches='tight')
    # plt.show()

    result_df = pd.DataFrame({
        'Time': time_sim,
        'X': X_sim,
        'cPDL': cPDL_sim,
        '95% CI': ci,
        'X_CI_upper': X_sim + ci,
        'X_CI_lower': X_sim - ci,
        '95% PI': pi,
        'X_PI_upper': X_sim + pi,
        'X_PI_lower': X_sim - pi
    })
    result_df.to_csv(csv_file3, index=False)

    def f_cPDL(cPDL, conc, lamda):
        mu_max = (conc - np.exp(cPDL * lamda))
        return mu_max

    cPDL_pred = np.linspace(0, 20, 101)
    mumax_pred = f_cPDL(cPDL_pred, *params)

    result_mu_df = pd.DataFrame({
        'cPDL': cPDL_pred,
        'mu_max': mumax_pred,
    })
    result_mu_df.to_csv(f'{header1}/results/fitted_mumax_CI_{n_dataset}.csv', index=False)

    n_dataset += 1