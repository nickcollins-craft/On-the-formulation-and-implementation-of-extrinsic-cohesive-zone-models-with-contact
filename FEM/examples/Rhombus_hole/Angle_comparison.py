import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

# Declare the name of the folder where the data is stored
data_folder = "/path/to/your/storage/folder/goes/here/"
# Declare the name of the folder where the image will be saved
save_folder = "/path/to/your/folder/goes/here/"
β_70 = np.load(data_folder + "70_degrees/beta.npy")
y_70 = np.load(data_folder + "70_degrees/y_pos.npy")
F_70 = np.load(data_folder + "70_degrees/initiation_force.npy")
β_80 = np.load(data_folder + "80_degrees/beta.npy")
y_80 = np.load(data_folder + "80_degrees/y_pos.npy")
F_80 = np.load(data_folder + "80_degrees/initiation_force.npy")
β_90 = np.load(data_folder + "90_degrees/beta.npy")
y_90 = np.load(data_folder + "90_degrees/y_pos.npy")
F_90 = np.load(data_folder + "90_degrees/initiation_force.npy")
β_100 = np.load(data_folder + "100_degrees/beta.npy")
y_100 = np.load(data_folder + "100_degrees/y_pos.npy")
F_100 = np.load(data_folder + "100_degrees/initiation_force.npy")
β_110 = np.load(data_folder + "110_degrees/beta.npy")
y_110 = np.load(data_folder + "110_degrees/y_pos.npy")
F_110 = np.load(data_folder + "110_degrees/initiation_force.npy")

# Sum up the nodal forces in F_vecs, and multiply by two to get the total force
# (given the symmetry)
F_stop_70 = 2*np.sum(F_70)
F_stop_80 = 2*np.sum(F_80)
F_stop_90 = 2*np.sum(F_90)
F_stop_100 = 2*np.sum(F_100)
F_stop_110 = 2*np.sum(F_110)

# Calculate the y-position in terms of distance from the initial crack
y_crack_70 = y_70 - 4.95
y_crack_80 = y_80 - 4.95
y_crack_90 = y_90 - 4.95
y_crack_100 = y_100 - 4.95
y_crack_110 = y_110 - 4.95

# Re-arrange the data so that it's in spatial order
sort_index_70 = np.argsort(y_crack_70)
sort_index_80 = np.argsort(y_crack_80)
sort_index_90 = np.argsort(y_crack_90)
sort_index_100 = np.argsort(y_crack_100)
sort_index_110 = np.argsort(y_crack_110)
β_70 = β_70[sort_index_70, :]
y_crack_70 = y_crack_70[sort_index_70]
β_80 = β_80[sort_index_80, :]
y_crack_80 = y_crack_80[sort_index_80]
β_90 = β_90[sort_index_90, :]
y_crack_90 = y_crack_90[sort_index_90]
β_100 = β_100[sort_index_100, :]
y_crack_100 = y_crack_100[sort_index_100]
β_110 = β_110[sort_index_110, :]
y_crack_110 = y_crack_110[sort_index_110]

# Find the last zero spatial index at the last temporal point
for node in range(len(y_crack_70)):
    if β_70[node, -1] <= 1e-12:
        L_70 = y_crack_70[node]
    else:
        break
for node in range(len(y_crack_80)):
    if β_80[node, -1] <= 1e-12:
        L_80 = y_crack_80[node]
    else:
        break
for node in range(len(y_crack_90)):
    if β_90[node, -1] <= 1e-12:
        L_90 = y_crack_90[node]
    else:
        break
for node in range(len(y_crack_100)):
    if β_100[node, -1] <= 1e-12:
        L_100 = y_crack_100[node]
    else:
        break
for node in range(len(y_crack_110)):
    if β_110[node, -1] <= 1e-12:
        L_110 = y_crack_110[node]
    else:
        break

# Now grab the data from Doitrand et al (manually specified)
doitrand_L_70 = 0.04563331604
doitrand_L_80 = 0.150063962046
doitrand_L_90 = 0.592645168304
doitrand_L_100 = 1.67044591904
doitrand_L_110 = 4.57458734512
doitrand_F_70 = -1363.38
doitrand_F_80 = -1065.09
doitrand_F_90 = -873.181
doitrand_F_100 = -728.45
doitrand_F_110 = -638.018

# Now make the vectors for plotting purposes
β_vec = [70, 80, 90, 100, 110]
L_vec = [L_70, L_80, L_90, L_100, L_110]
F_vec = [F_stop_70, F_stop_80, F_stop_90, F_stop_100, F_stop_110]
doitrand_L_vec = [doitrand_L_70, doitrand_L_80, doitrand_L_90, doitrand_L_100, doitrand_L_110]
# Multiply force values by 10 to reflect true thickness
doitrand_F_vec = [10*doitrand_F_70, 10*doitrand_F_80, 10*doitrand_F_90, 10*doitrand_F_100, 10*doitrand_F_110]

# Plot the comparisons
width_in_inches = (8.27-2*1.5/2.54)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(width_in_inches, 0.5*0.625*width_in_inches))
colour_split = plt.cm.viridis(np.linspace(0, 1, 5))
marker_split = ['x', 'd']
# TeX the written elements so that it looks good (comment out until final run
# because calling TeX is *slow*)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Comparison plot
ax0 = plt.subplot(121)
ax0.set_prop_cycle('color', colour_split)
ax0.scatter(β_vec, np.abs(F_vec),  marker=marker_split[0])
ax0.scatter(β_vec, np.abs(doitrand_F_vec), marker=marker_split[1])
ax0.set_xlabel(r'Rhombus hole angle ($^{\circ}$)', fontsize=10)
ax0.set_ylabel(r'$F_{\mathrm{crack}}$ (N)', fontsize=10)
plt.xlim(65, 115)
plt.ylim(0, 18000)
ax0.minorticks_on()
ax0.tick_params(axis='x', which='minor', bottom=False)
ax0.tick_params(labelsize=10)
text_x_pos = 0.015
text_y_pos = 0.90
ax0.text(text_x_pos, text_y_pos, '$(a)$', transform=ax0.transAxes, fontsize=10)

ax1 = plt.subplot(122)
ax1.set_prop_cycle('color', colour_split)
ax1.scatter(β_vec, L_vec,  marker=marker_split[0], label=r'Extrinsic CZM')
ax1.scatter(β_vec, doitrand_L_vec, marker=marker_split[1], label=r'Doitrand et al.')
ax1.set_xlabel(r'Rhombus hole angle ($^{\circ}$)', fontsize=10)
ax1.set_ylabel(r'$\mathcal{L}_{\mathrm{arrest}}$ (mm)', fontsize=10)
plt.xlim(65, 115)
plt.ylim(-0.2, 4.9)
ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1.tick_params(labelsize=10)
ax1.text(text_x_pos, text_y_pos, '$(b)$', transform=ax1.transAxes, fontsize=10)

plt.tight_layout()

# Make the legend
fig.subplots_adjust(top=0.88)
fig.legend(loc=9, ncol=2, borderaxespad=0.)

plt.savefig(save_folder + "Figure_18.pdf")

plt.show()
