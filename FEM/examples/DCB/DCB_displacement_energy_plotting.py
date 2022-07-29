import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

# Declare the name of the folder where the data is stored
data_folder = "/path/to/your/storage/folder/goes/here/"
# Declare the name of the folder where the image will be saved
save_folder = "/path/to/your/folder/goes/here/"
Ψ_ε = np.load(data_folder + "strain_energy.npy")
K = np.load(data_folder + "kinetic_energy.npy")
Ψ_s = np.load(data_folder + "surface_energy.npy")
W = np.load(data_folder + "work.npy")
W_p = np.load(data_folder + "contact_work.npy")
t = np.load(data_folder + "t.npy")

# Calculate the total energy
total_energy = Ψ_ε + K + Ψ_s

# Get the maximum time
tmax = t[-1]

# Plot time!
width_in_inches = (8.27-2*1.5/2.54)/1.5
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(width_in_inches, 0.625*width_in_inches))
dof_index = [0, 1, 2, 3, 4]
colour_split = plt.cm.viridis(np.linspace(0, 1, 5))
line_split = ['-', '--', '-.', ':', '-', '--']
# TeX the written elements so that it looks good (comment out until final run
# because calling TeX is *slow*)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


ax0 = plt.subplot(111)
ax0.set_prop_cycle('color', colour_split)
ax0.plot(t[:-1], Ψ_ε, linestyle=line_split[0], linewidth=1, label=r'Strain energy')
ax0.plot(t[:-1], K, linestyle=line_split[1], linewidth=1, label=r'Kinetic energy')
ax0.plot(t[:-1], Ψ_s, linestyle=line_split[2], linewidth=1, label=r'Surface energy')
ax0.plot(t[:-1], W, color=colour_split[4], linestyle=line_split[4], linewidth=1, label=r'Total work')
ax0.plot(t[:-1], total_energy, color=colour_split[3], linestyle=line_split[3], linewidth=1, label=r'Total energy') # Plot them in order of W - total energy so they're easier to see
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'Energy (N$\cdot$mm)', fontsize=10)
plt.xlim(0, tmax)
ax0.minorticks_on()
ax0.tick_params(labelsize=10)
plt.legend(loc=0, fontsize=10)

# Tighten the layout
plt.tight_layout()

# Save it
plt.savefig(save_folder + "Figre_11.pdf")

plt.show()
