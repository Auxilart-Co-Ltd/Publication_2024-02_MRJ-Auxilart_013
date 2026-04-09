import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 20days cultivation, varying btb day and passage number
cell_labels = ['8580', '2827', '8331']
# cell_labels = ['8580']
cell_label = '8331'
p_labels = [3, 4]

outdir = './dataout/sensitivity'
infile_csv = f'{outdir}/sensitivity_ds_08e8_{cell_label}.csv'
result_df = pd.read_csv(infile_csv)

x = result_df['Passage'].to_numpy()
y = result_df['t_btb1_days'].to_numpy()
z = result_df['Binary_Total_n_cells'].to_numpy()
z1 = result_df['Binary_X_btb1_cells_per_mL'].to_numpy()
z2 = result_df['Binary_X_btb2_cells_per_mL'].to_numpy()
z3 = result_df['Binary_X_end_cells_per_mL'].to_numpy()

z4 = result_df['Binary_ds'].to_numpy()
n_passages = len(np.unique(x))
n_tbtb1 = len(np.unique(y))

x = x.reshape((n_passages, n_tbtb1))
y = y.reshape((n_passages, n_tbtb1))
z = z.reshape((n_passages, n_tbtb1))
z1 = z1.reshape((n_passages, n_tbtb1))
z2 = z2.reshape((n_passages, n_tbtb1))
z3 = z3.reshape((n_passages, n_tbtb1))
z4 = z4.reshape((n_passages, n_tbtb1))

plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

from matplotlib.colors import ListedColormap
vmin = 0
vmax = 1

binary_cmap = ListedColormap([
    (0.3, 0.3, 0.3, 0.6),
    (1.0, 1.0, 1.0, 0.0)
])
binary_cmap_orange = ListedColormap([
    (104/256, 197/256, 219/256, 0.8),
    (1.0, 1.0, 1.0, 0.0)
])

extent = [x.min() - 0.5, x.max() + 0.5,
        y.min() - 0.5, y.max() + 0.5]

hm0 = axes[0, 0].imshow(z.T, cmap=binary_cmap_orange, origin='lower',
                        extent=extent, vmin=vmin, vmax=vmax,
                        aspect='auto')
cbar0 = fig.colorbar(hm0, ax=axes[0, 0])
cbar0.set_label('Total number of cells cultivated')
axes[0, 0].set_ylabel('Culture interval until first BtB [days]')
axes[0, 0].set_xlabel('Passage number')
axes[0, 0].set_title('Total cells')
levels = [0.5]
fac = 10
condition = np.kron(z > 0.8, np.ones((fac, fac)))

# X_btb1
hm1 = axes[0, 1].imshow(z1.T, cmap=binary_cmap, origin='lower',
                        extent=extent, vmin=vmin, vmax=vmax,
                        aspect='auto')
cbar1 = fig.colorbar(hm1, ax=axes[0, 1])
cbar1.set_label('cells per mL')
axes[0, 1].set_ylabel('Culture interval until first BtB [days]')
axes[0, 1].set_xlabel('Passage number')
axes[0, 1].set_title('X_btb1_cells_per_mL')

# X_btb2
hm2 = axes[1, 0].imshow(z2.T, cmap=binary_cmap, origin='lower',
                        extent=extent, vmin=vmin, vmax=vmax,
                        aspect='auto')
cbar2 = fig.colorbar(hm2, ax=axes[1, 0])
cbar2.set_label('cells per mL')
axes[1, 0].set_ylabel('Culture interval until first BtB [days]')
axes[1, 0].set_xlabel('Passage number')
axes[1, 0].set_title('X_btb2_cells_per_mL')

# X_end
hm3 = axes[1, 1].imshow(z3.T, cmap=binary_cmap, origin='lower',
                        extent=extent, vmin=vmin, vmax=vmax,
                        aspect='auto')
cbar3 = fig.colorbar(hm3, ax=axes[1, 1])
cbar3.set_label('cells per mL')
axes[1, 1].set_ylabel('Culture interval until first BtB [days]')
axes[1, 1].set_xlabel('Passage number')
axes[1, 1].set_title('X_end_cells_per_mL')

fig.suptitle(f'Sensitivity Analysis for Cell Line {cell_label})')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.figure(figsize=(8,6))
hm4 = plt.imshow(z.T, cmap=binary_cmap_orange, origin='lower',
                extent=extent, vmin=vmin, vmax=vmax, aspect='auto')
hm4 = plt.imshow(z1.T, cmap=binary_cmap, origin='lower',
                extent=extent, vmin=vmin, vmax=vmax, aspect='auto')
hm4 = plt.imshow(z2.T, cmap=binary_cmap, origin='lower',
                extent=extent, vmin=vmin, vmax=vmax, aspect='auto')
hm4 = plt.imshow(z3.T, cmap=binary_cmap, origin='lower',
                extent=extent, vmin=vmin, vmax=vmax, aspect='auto')
condition = np.kron(z4 > 0.8, np.ones((fac, fac)))
cbar4 = plt.colorbar(hm4)
if cell_label == '8580':
    plt.xticks([3,4,5,6,7,8,9,10])
elif cell_label == '2827':
    plt.xticks([3,4,5,6,7,8,9])
elif cell_label == '8331':
    plt.xticks(np.arange(3, 14, 1))
plt.yticks(np.arange(3,16, 2))
plt.ylabel('Cultivation time [days]')
plt.xlabel('Passage number')
plt.savefig(f'./fig/design_space/design_space_{cell_label}.png', dpi=400, bbox_inches='tight')
plt.show()
