import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.signal import savgol_filter

# Declare the name of the folder where the data is stored
data_folder = "/path/to/your/storage/folder/goes/here/"
# Declare the name of the folder where the image will be saved
save_folder = "/path/to/your/folder/goes/here/"
t = np.load(data_folder + "t.npy")
β = np.load(data_folder + "beta.npy")
y = np.load(data_folder + "y_pos.npy")

# Get the maximum time
tmax = t[-1]

# Calculate the x-position in terms of distance from the initial crack
y_crack = y - 4.95
# Re-arrange the data so that it's in spatial order
y_order = np.argsort(y_crack)
y_crack = y_crack[y_order]
β = β[y_order]

# As well as where each point goes to zero (starting their value higher than the
# last step)
t_step_zeros = (len(t) + 1)*np.ones((np.shape(β)[0]), dtype=int)
zeros_truth = np.ones((np.shape(β)[0]), dtype=bool)
for step_number in range(len(t)):
    for index in range(np.shape(β)[0]):
        if zeros_truth[index]:
            if β[index, step_number] <= 1e-12:
                t_step_zeros[index] = step_number
                zeros_truth[index] = False

# Calculate the quarter points (spatial), depending on the last one to crack
index = 0
while zeros_truth[index] == False:
    last_cracked = index
    index = index + 1

q1_index = int(np.round(last_cracked/4))
q2_index = int(np.round(last_cracked/2))
q3_index = int(np.round(3*last_cracked/4))
# Put them in a group for plotting and analysis
β_dofs = np.array([β[0, :], β[q1_index, :], β[q2_index, :], β[q3_index, :], β[last_cracked, :]])

# Calculate the crack length and velocity, and the time behaviour of the quarter points
L = np.zeros((len(t)))
t_quarters = np.zeros((5), dtype=int)
t_quarter_truths = np.ones((5), dtype=bool)
node_index = 0
for step_number in range(len(t)):
    if step_number < t_step_zeros[node_index]:
        L[step_number] = L[step_number - 1]
    else:
        L[step_number] = L[0] + y_crack[node_index]
        node_index = node_index + 1
    for i in range(5):
        if t_quarter_truths[i]:
            if β_dofs[i, step_number] <= 1e-12:
                t_quarter_truths[i] = False
                t_quarters[i] = step_number

# Smooth with a Savitzky-Golay filter
L_smoothed = savgol_filter(L, 31, 1)

# Now do the crack velocities, linearly interpolating the crack length between
# the points where β goes to 0, and thus giving a constant velocity
L_dot = np.zeros_like(L)
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
E = 1600
ν = 0.37
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
ax0.set_xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$\beta$', fontsize=10)
plt.xlim(0, 1.01*tmax)
plt.ylim(0.0, 1.2)
ax0.minorticks_on()
ax0.tick_params(labelsize=10)
plt.legend(loc=6, ncol=3, borderaxespad=0., bbox_to_anchor=(0.08, 1.19))
text_x_pos = 0.015
text_y_pos = 0.90
ax0.text(text_x_pos, text_y_pos, '$(a)$', transform=ax0.transAxes, fontsize=10)
# Create an inset axis to show the crack progression over the relevant time-scale
ax0_inset = inset_axes(ax0, width="100%", height="100%", bbox_to_anchor=(.075, .15, .525, .55), bbox_transform=ax0.transAxes, loc=3)
ax0_inset.set_prop_cycle('color', colour_split)
for index in dof_index:
    ax0_inset.plot(t, β_dofs[index, :], linestyle=line_split[index], linewidth=1)
inset_left_xlim = 448.0
plt.xlim(inset_left_xlim, tmax)
plt.ylim(0.0, 1.1)
ax0_inset.minorticks_on()
ax0_inset.tick_params(labelsize=8)
mark_inset(ax0, ax0_inset, loc1=1, loc2=4, ec="0.8")

# β in space as each quarter point goes to zero
ax1 = plt.subplot(222)
ax1.set_prop_cycle('color', colour_split)
for index in dof_index:
    ax1.plot(y_crack, β[:, t_quarters[index]], linestyle=line_split[index], linewidth=1, label=r'$t_{}$'.format(index))
plt.xlabel(r'$x_{\mathrm{crack}}$ (mm)', fontsize=10)
plt.ylabel(r'$\beta$', fontsize=10)
#plt.xlim(0, y_crack[last_cracked + 50])
plt.xlim(0, 0.8)
plt.ylim(0, 1.2)
ax1.minorticks_on()
ax1.tick_params(labelsize=10)
plt.legend(loc=6, ncol=3, borderaxespad=0., bbox_to_anchor=(0.08, 1.19))
ax1.text(text_x_pos, text_y_pos, '$(b)$', transform=ax1.transAxes, fontsize=10)

# Crack length with time
ax2 = plt.subplot(223)
ax2.set_prop_cycle('color', colour_split)
ax2.plot(t, L, linestyle=line_split[index], linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$\mathcal{L}$ (mm)', fontsize=10)
plt.xlim(0.0, 1.01*tmax)
plt.ylim(-0.01, 1.1*np.max(L))
ax2.minorticks_on()
ax2.tick_params(labelsize=10)
ax2.text(text_x_pos, text_y_pos, '$(c)$', transform=ax2.transAxes, fontsize=10)
# Create an inset axis to show the crack progression over the relevant time-scale
ax2_inset = inset_axes(ax2, width="100%", height="100%", bbox_to_anchor=(.15, .175, .7, .725), bbox_transform=ax2.transAxes, loc=3)
ax2_inset.set_prop_cycle('color', colour_split)
ax2_inset.plot(t, L, linestyle=line_split[index], linewidth=1)
plt.xlim(inset_left_xlim, tmax)
plt.ylim(-0.01, 1.05*np.max(L))
ax2_inset.minorticks_on()
ax2_inset.tick_params(labelsize=8)
mark_inset(ax2, ax2_inset, loc1=1, loc2=4, ec="0.8")

# Crack velocity with time
ax3 = plt.subplot(224)
ax3.set_prop_cycle('color', colour_split)
ax3.plot(t, L_dot_normalised, linestyle=line_split[index], linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$\dot{\mathcal{L}}/c_{\mathrm{R}}$', fontsize=10)
plt.xlim(0.0, 1.01*tmax)
plt.ylim(-0.001, 1.1*np.max(L_dot_normalised))
ax3.minorticks_on()
ax3.tick_params(labelsize=10)
ax3.text(text_x_pos, text_y_pos, '$(d)$', transform=ax3.transAxes, fontsize=10)
# Create an inset axis to show the crack progression over the relevant time-scale
ax3_inset = inset_axes(ax3, width="100%", height="100%", bbox_to_anchor=(.175, .175, .725, .725), bbox_transform=ax3.transAxes, loc=3)
ax3_inset.set_prop_cycle('color', colour_split)
ax3_inset.plot(t, L_dot_normalised, linestyle=line_split[index], linewidth=1)
plt.xlim(inset_left_xlim, tmax)
plt.ylim(-0.001, 1.05*np.max(L_dot_normalised))
ax3_inset.minorticks_on()
ax3_inset.tick_params(labelsize=8)
mark_inset(ax3, ax3_inset, loc1=1, loc2=4, ec="0.8")

# Tighten the layout
plt.tight_layout()

# Save it
plt.savefig(save_folder + "Figure_16.pdf")

plt.show()
