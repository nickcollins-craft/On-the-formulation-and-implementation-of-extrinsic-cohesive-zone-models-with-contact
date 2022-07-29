import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter

# Declare the name of the folder where the data is stored
data_folder = "/path/to/your/storage/folder/goes/here/"
# Declare the name of the folder where the image will be saved
save_folder = "/path/to/your/folder/goes/here/"
t = np.load(data_folder + "t.npy")
β = np.load(data_folder + "beta.npy")
x = np.load(data_folder + "x_pos.npy")

# Get the maximum time
tmax = t[-1]

# Calculate the x-position in terms of distance from the initial crack
x_crack = 1.6 - x
# Re-arrange the data so that it's in spatial order
β = np.concatenate((β[1:, :], np.array(β[0, :], ndmin=2)), axis=0)
x_crack = np.concatenate((x_crack[1:], np.array(x_crack[0], ndmin=1)))

# Calculate the quarter points (spatial)
q1 = int(np.round(np.shape(β)[0]/4))
q2 = int(np.round(np.shape(β)[0]/2))
q3 = int(np.round(3*np.shape(β)[0]/4))

β_dofs = np.array([β[-1, :], β[q3, :], β[q2, :], β[q1, :], β[0, :]])

# Calculate the points when each of the quarter points goes to zero
t_quarters = np.zeros((5), dtype=int)
t_quarter_truths = np.ones((5), dtype=bool)
# As well as where each point goes to zero
t_step_zeros = len(t)*np.ones((np.shape(β)[0]), dtype=int)
zeros_truth = np.ones((np.shape(β)[0]), dtype=bool)
for step_number in range(len(t)):
    for i in range(5):
        if t_quarter_truths[i]:
            if β_dofs[i, step_number] <= 1e-12:
                t_quarter_truths[i] = False
                t_quarters[i] = step_number
    for index in range(np.shape(β)[0]):
        if zeros_truth[index]:
            if β[index, step_number] <= 1e-12:
                t_step_zeros[index] = step_number
                zeros_truth[index] = False

# Calculate the crack length and velocity
L = 0.4*np.ones((len(t)))
sorted_x_crack = np.sort(x_crack)
sorted_t_step_zeros = t_step_zeros[np.argsort(x_crack)]
search_index = 0
for step_number in range(len(t)):
    if step_number < sorted_t_step_zeros[search_index]:
        L[step_number] = L[step_number - 1]
    else:
        trial_search_index = search_index + 1
        while sorted_t_step_zeros[trial_search_index] == sorted_t_step_zeros[search_index]:
            trial_search_index = trial_search_index + 1
            L[step_number] = L[0] + sorted_x_crack[trial_search_index]
        else:
            L[step_number] = L[0] + sorted_x_crack[search_index]
            search_index = trial_search_index

# Smooth with a Savitzky-Golay filter
L_smoothed = savgol_filter(L, 31, 1)
# Now do the crack velocities, linearly interpolating the crack length between
# the points where β goes to 0, and thus giving a constant velocity
L_dot = np.zeros(len(L))
arranged_zero_steps = np.sort(t_step_zeros)
search_index = 1
for step_number in range(len(t)):
    if step_number <= arranged_zero_steps[0]:
        L_dot[step_number] = 0.0
    elif step_number <= arranged_zero_steps[search_index]:
        if arranged_zero_steps[search_index] >= len(L) - 1:
            L_dot[step_number] = 0.0
        else:
            trial_search_index = search_index
            while t[arranged_zero_steps[trial_search_index]] == t[arranged_zero_steps[search_index - 1]]:
                trial_search_index = trial_search_index + 1
                L_dot[step_number] = (L_smoothed[arranged_zero_steps[trial_search_index]] - L_smoothed[[arranged_zero_steps[search_index - 1]]])/(t[arranged_zero_steps[trial_search_index]] - t[arranged_zero_steps[search_index - 1]])
            else:
                search_index = trial_search_index
                L_dot[step_number] = (L_smoothed[arranged_zero_steps[search_index]] - L_smoothed[[arranged_zero_steps[search_index - 1]]])/(t[arranged_zero_steps[search_index]] - t[arranged_zero_steps[search_index - 1]])
    else:
        search_index = search_index + 1
        if arranged_zero_steps[search_index] >= len(L) - 1:
            L_dot[step_number] = 0.0
        else:
            trial_search_index = search_index
            while t[arranged_zero_steps[trial_search_index]] == t[arranged_zero_steps[search_index - 1]]:
                trial_search_index = trial_search_index + 1
                L_dot[step_number] = (L_smoothed[arranged_zero_steps[trial_search_index]] - L_smoothed[[arranged_zero_steps[search_index - 1]]])/(t[arranged_zero_steps[trial_search_index]] - t[arranged_zero_steps[search_index - 1]])
            else:
                search_index = trial_search_index
                L_dot[step_number] = (L_smoothed[arranged_zero_steps[search_index]] - L_smoothed[[arranged_zero_steps[search_index - 1]]])/(t[arranged_zero_steps[search_index]] - t[arranged_zero_steps[search_index - 1]])


# Calculate the shear wave speed from the elastic properties
E = 2.7E3
ν = 0.39
ρ = 1.18E-3
# First get G, shear stiffness
G = E/(2*(1 + ν))
# Now the shear wave speed
c_s = np.sqrt(G/ρ)
# Now empirical formula for the Rayleigh wave speed
c_r = c_s*(0.862 + 1.14*ν)/(1 + ν)
# Divide the speed by c_r
L_dot_normalised = L_dot/c_r


# Plot time!
width_in_inches = (8.27-2*1.5/2.54)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(width_in_inches, 0.625*width_in_inches))
dof_index = [0, 1, 2, 3, 4]
colour_split = plt.cm.viridis(np.linspace(0, 1, 5))
line_split = ['-', '--', '-.', ':', '-', '--']
# TeX the written elements so that it looks good (comment out until final run
# because calling TeX is *slow*)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# β at the quarter points in time
ax0 = plt.subplot(221)
ax0.set_prop_cycle('color', colour_split)
for index in dof_index:
    ax0.plot(t, β_dofs[index, :], linestyle=line_split[index], linewidth=1, label=r'$\beta_{}$'.format(index))
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$\beta$', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(0.0, 1.2)
ax0.minorticks_on()
ax0.tick_params(labelsize=10)
plt.legend(loc=6, ncol=3, borderaxespad=0., bbox_to_anchor=(0.08, 1.19))
text_x_pos = 0.0075
text_y_pos = 0.90
ax0.text(text_x_pos, text_y_pos, '$(a)$', transform=ax0.transAxes, fontsize=10)

# β in space as each quarter point goes to zero
ax1 = plt.subplot(222)
ax1.set_prop_cycle('color', colour_split)
for index in dof_index:
    ax1.plot(x_crack, β[:, t_quarters[index]], linestyle=line_split[index], linewidth=1, label=r'$t_{}$'.format(index))
plt.xlabel(r'$x_{\mathrm{crack}}$ (mm)', fontsize=10)
plt.ylabel(r'$\beta$', fontsize=10)
plt.xlim(0, np.max(x_crack))
plt.ylim(0, 1.2)
ax1.minorticks_on()
ax1.tick_params(labelsize=10)
plt.legend(loc=6, ncol=3, borderaxespad=0., bbox_to_anchor=(0.08, 1.19))
ax1.text(text_x_pos, text_y_pos, '$(b)$', transform=ax1.transAxes, fontsize=10)

# Crack length with time
ax2 = plt.subplot(223)
ax2.set_prop_cycle('color', colour_split)
ax2.plot(t, L_smoothed, linestyle=line_split[index], linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$\mathcal{L}$ (mm)', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(0.0, 1.1*np.max(L))
ax2.minorticks_on()
ax2.tick_params(labelsize=10)
ax2.text(text_x_pos, text_y_pos, '$(c)$', transform=ax2.transAxes, fontsize=10)

# Crack velocity with time
ax3 = plt.subplot(224)
ax3.set_prop_cycle('color', colour_split)
ax3.plot(t, L_dot_normalised, linestyle=line_split[index], linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$\dot{\mathcal{L}}/c_{_{\mathrm{R}}}$', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(np.min(L_dot_normalised), 1.1*np.max(L_dot_normalised))
ax3.minorticks_on()
ax3.tick_params(labelsize=10)
ax3.text(text_x_pos, text_y_pos, '$(d)$', transform=ax3.transAxes, fontsize=10)

# Tighten the layout
plt.tight_layout()

# Save it
plt.savefig(save_folder + "Figure_12.pdf")

plt.show()
