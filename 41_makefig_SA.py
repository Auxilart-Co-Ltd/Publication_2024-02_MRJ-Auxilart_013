import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

cell_labels = ['8580', '2827', '8331']
cell_label = '8331'

if cell_label == '8580':
    p_end = 10
elif cell_label == '2827':
    p_end = 9
elif cell_label == '8331':
    p_end = 13

# save to outfiles
outdir = './fig/design_space'
os.makedirs(outdir, exist_ok=True)
outfile_png1 = f'{outdir}/{cell_label}_total_cells.png'
outfile_png2 = f'{outdir}/{cell_label}_confluency_btb1.png'
outfile_png3 = f'{outdir}/{cell_label}_confluency_btb2.png'
outfile_png4 = f'{outdir}/{cell_label}_confluency_3.png'

# for cell_label in cell_labels:
outdir = './dataout/sensitivity'
infile_csv = f'{outdir}/sensitivity_tbtb1_p_{cell_label}.csv'
result_df = pd.read_csv(infile_csv)

x = result_df['Passage'].to_numpy()
y = result_df['t_btb1_days'].to_numpy()
z = result_df['Total_n_cells'].to_numpy()
z1 = result_df['X_btb1_cells_per_mL'].to_numpy()
z2 = result_df['X_btb2_cells_per_mL'].to_numpy()
z3 = result_df['X_end_cells_per_mL'].to_numpy()

# reshape
n_passages = len(np.unique(x))
n_tbtb1 = len(np.unique(y))

x = x.reshape((n_passages, n_tbtb1))
y = y.reshape((n_passages, n_tbtb1))
z = z.reshape((n_passages, n_tbtb1))
z1 = z1.reshape((n_passages, n_tbtb1))
z2 = z2.reshape((n_passages, n_tbtb1))
z3 = z3.reshape((n_passages, n_tbtb1))
# print(x)

X_max = 3e5
P_btb1 = z1 / X_max
P_btb2 = z2 / X_max
P_3 = z3 / X_max

cmap = 'YlGnBu'
cmap_X = 'YlOrRd'
extent = [x.min()-0.5, x.max()+0.5, y.min()-0.5, y.max()+0.5]

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.figure(figsize=(8, 6))
hm0 = plt.imshow(z.T*1e-8, cmap=cmap, origin='lower', extent=extent, aspect='auto', vmin=0, vmax=1.6)
plt.colorbar(hm0, label = 'Total number of cells cultivated [1e8]')

threshold = 0.8e8
fac = 10
condition = np.kron(z > threshold, np.ones((fac, fac)))
plt.contour(condition.T, levels=[0.5], extent=extent, colors='red', linewidths=2)

plt.xlabel('Passage number')
plt.xticks(np.arange(3, p_end+1, 1))
plt.yticks(np.arange(3,16, 2))
plt.ylabel('Culture interval until first BtB [days]')
plt.savefig(outfile_png1, dpi=400, bbox_inches='tight')
plt.show()

# X_btb1
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.figure(figsize=(8, 6))
hm0 = plt.imshow(P_btb1.T, cmap=cmap_X, origin='lower', extent=extent, aspect='auto', vmin=0, vmax=1.0)
plt.colorbar(hm0, label = 'Confluency level')

threshold = 0.8
fac = 10
condition = np.kron(P_btb1 > threshold, np.ones((fac, fac)))
plt.contour(condition.T, levels=[0.5], extent=extent, colors='blue', linewidths=2)

plt.xlabel('Passage number')
plt.xticks(np.arange(3, p_end+1, 1))
plt.yticks(np.arange(3,16, 2))
plt.ylabel('Culture interval until first BtB [days]')
plt.savefig(outfile_png2, dpi=400, bbox_inches='tight')
plt.show()

# X_btb2
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.figure(figsize=(8, 6))
hm0 = plt.imshow(P_btb2.T, cmap=cmap_X, origin='lower', extent=extent, aspect='auto', vmin=0, vmax=1.0)
plt.colorbar(hm0, label = 'Confluency level')

threshold = 0.8
fac = 10
condition = np.kron(P_btb2 > threshold, np.ones((fac, fac)))
plt.contour(condition.T, levels=[0.5], extent=extent, colors='blue', linewidths=2)

plt.xlabel('Passage number')
plt.xticks(np.arange(3, p_end+1, 1))
plt.yticks(np.arange(3,16, 2))
plt.ylabel('Culture interval until first BtB [days]')
plt.savefig(outfile_png3, dpi=400, bbox_inches='tight')
plt.show()

# X_3
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.figure(figsize=(8, 6))
hm0 = plt.imshow(P_3.T, cmap=cmap_X, origin='lower', extent=extent, aspect='auto', vmin=0, vmax=1.0)
plt.colorbar(hm0, label = 'Confluency level')

threshold = 0.8
fac = 10
condition = np.kron(P_3 > threshold, np.ones((fac, fac)))
plt.contour(condition.T, levels=[0.5], extent=extent, colors='blue', linewidths=2)

plt.xlabel('Passage number')
plt.xticks(np.arange(3, p_end+1, 1))
plt.yticks(np.arange(3,16, 2))
plt.ylabel('Culture interval until first BtB [days]')
plt.savefig(outfile_png4, dpi=400, bbox_inches='tight')
plt.show()