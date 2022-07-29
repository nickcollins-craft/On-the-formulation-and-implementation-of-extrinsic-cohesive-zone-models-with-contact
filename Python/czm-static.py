import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import siconos.numerics as sn
import siconos

# Declare the name of the folder to save the figure
save_folder = "/path/to/your/folder/goes/here/"

# Material constants
# Critical traction
σ_c = 1/2
# Critical opening length
δ_c = 1

# Number of time-steps
n = 1000
# Maximum integration time
tmax = 6.0

# We can set the Siconos LCP solver here. Lemke is a direct method using pivoting,
# PGS (Projective Gauss-Seidel) is an iterative solver
id = sn.SICONOS_LCP_LEMKE
#id = sn.SICONOS_LCP_PGS

# Initial values of intactness and λ and μ complemtarity variables, and thermodynamic force "A"
β = [1.]
λ = [0.]
μ = [0.]
A = [0.]

# Define a loading protocol (load-unload-reload)
def load_0(t):
    if t <= 1.0:
        u = 0.5*t
    elif 1.0 < t <= 2.0:
        u = 1 - 0.5*t
    elif t > 2.0:
        u  = - 1 + 0.5*t
    return u


# Define the load
load = load_0

# Create the time steps and print them
t = np.arange(0, tmax + tmax/n, tmax/n)
print(t)

# Initialise the driving displacement as the first value of the load
u = [load(0)]

# Define the test, inputting the initial values of the intactness and the
# complementarity variables
def test(β, A, u, λ, μ):
    # Let it know that it's an LCP of size 2
    lcp_size = 2
    # For each time step
    for k in range(n):
        # Print the step
        print (k)
        # Append the output of the load function to the driving displacement, and print it
        u.append(load(t[k]))
        print('u', u[-1])

        # Define the problem.
        L = np.array([[0, -1], [1, σ_c*δ_c]])
        q = np.array([β[-1], -σ_c*(δ_c*(β[-1] - 1) + u[-1])])
        # Tell Siconos how to understand the problem (i.e. that it's an LCP)
        lcp = sn.LCP(L, q)

        # Declare two zero vectors that will hold the solutions of the LCP. These
        # are vectors that solve the system Lz + q = w, where z and w are normal.
        z = np.zeros((lcp_size,), np.float64)
        w = np.zeros_like(z)
        # Extract the solver options from Siconos given the type of LCP algorithm
        options = sn.SolverOptions(id)
        # Run the linear complementarity solver, given the LCP, the solution vectors and the options
        info = sn.linearComplementarity_driver(lcp, z, w, options)
        # Print the iterations and the solution residuals
        print('iter = ', options.iparam[sn.SICONOS_IPARAM_ITER_DONE])
        print('error = ', options.dparam[sn.SICONOS_DPARAM_RESIDU])
        # Print the solutions
        print(w, z)

        # Append the value of β with the the most recent value of β, w[0]
        β.append(w[0])
        # Append the value of A with the most recent value of A, w[1]
        A.append(w[1])
        # Append the value of μ with the most recent value of μ, z[0]
        μ.append(z[0])
        # Append the value of λ with the most recent value of λ, z[1] divided by the time-step
        λ_val = z[1]/(tmax/n)
        λ.append(λ_val)

        # Print the value of β
        print('β = ', β[-1])
    return

# Run the test, which lets us compare our LCP solution with the analytical solution
test(β, A, u, λ, μ)

# Let's write the analytic solutions to the variables of interest, per the paper
β_analytic = np.zeros_like(u)
A_analytic = np.zeros_like(u)
μ_analytic = np.zeros_like(u)
λ_analytic = np.zeros_like(u)
for (τ, time_step) in zip(t, range(1, len(t))):
    if τ <= 1.0:
        β_analytic[time_step] = 1 - u[time_step]/δ_c
        A_analytic[time_step] = 0.0
        μ_analytic[time_step] = 0.0
        λ_analytic[time_step] = (load_0(1) - load_0(0))/(1.0 - 0.0)*(1/δ_c)
    elif τ <= 2.0:
        β_analytic[time_step] = β_analytic[time_step - 1]
        A_analytic[time_step] = (σ_c/2)*τ + (σ_c*δ_c/2) - σ_c
        μ_analytic[time_step] = 0.0
        λ_analytic[time_step] = 0.0
    elif τ <= 3.0:
        β_analytic[time_step] = β_analytic[time_step - 1]
        A_analytic[time_step] = -(σ_c/2)*τ - σ_c*δ_c*(β_analytic[time_step] - 1) + σ_c
        μ_analytic[time_step] = 0.0
        λ_analytic[time_step] = 0.0
    elif τ <= 4.0:
        β_analytic[time_step] = (1/(σ_c*δ_c))*(σ_c*δ_c + σ_c - (σ_c/2)*τ)
        A_analytic[time_step] = 0.0
        μ_analytic[time_step] = 0.0
        λ_analytic[time_step] = (load_0(4) - load_0(3))/(4.0 - 3.0)*(1/δ_c)
    else:
        β_analytic[time_step] = β_analytic[time_step - 1]
        A_analytic[time_step] = 0.0
        μ_analytic[time_step] = (σ_c/2)*τ - σ_c - σ_c*δ_c
        λ_analytic[time_step] = 0.0



# Time to plot!
# Set the width in inches of the desired plot (everything in the brackets is
# basically full width of an A4 page with 1.5 cm margins)
width_in_inches = (8.27-2*1.5/2.54)/1
# Create the figure with the requisite number of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(width_in_inches, 0.625*width_in_inches))
# TeX the written elements so that it looks good (comment out until final run
# because calling TeX is *slow*)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
colour_split = plt.cm.viridis(np.linspace(0, 1, 5))

# Normal displacement with time
ax0 = plt.subplot(231)
ax0.set_prop_cycle('color', colour_split)
ax0.plot(t, u, linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$u_{\hbox{\tiny{N}}}$ (mm)', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(0, np.max(u))
ax0.minorticks_on()
ax0.tick_params(labelsize=10)
text_x_pos = 0.02
text_y_pos = 0.925
ax0.text(text_x_pos, text_y_pos, '$(a)$', transform=ax0.transAxes, fontsize=10)

# Intactness with time
ax1 = plt.subplot(232)
ax1.set_prop_cycle('color', colour_split)
#ax1.plot(t, β, linewidth=1, label="LCP")
ax1.plot(t, β_analytic, linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$\beta$', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(0, 1.1)
ax1.minorticks_on()
ax1.tick_params(labelsize=10)
ax1.text(text_x_pos, text_y_pos, '$(b)$', transform=ax1.transAxes, fontsize=10)

# Intactness with displacement
ax2 = plt.subplot(233)
ax2.set_prop_cycle('color', colour_split)
#ax2.plot(u, β, linewidth=1)
ax2.plot(u, β_analytic, linewidth=1)
plt.xlabel(r'$u_{\hbox{\tiny{N}}}$ (mm)', fontsize=10)
plt.ylabel(r'$\beta$', fontsize=10)
plt.xlim(0, np.max(u))
plt.ylim(0, 1.1)
ax2.minorticks_on()
ax2.tick_params(labelsize=10)
ax2.text(text_x_pos, text_y_pos, '$(c)$', transform=ax2.transAxes, fontsize=10)

# Thermodynamic force A
ax3 = plt.subplot(234)
ax3.set_prop_cycle('color', colour_split)
#ax4.plot(t, A, linewidth=1)
ax3.plot(t, A_analytic, linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$A^{\sf r}$ (N/mm)', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(0, 1.05*np.max(A_analytic))
ax3.minorticks_on()
ax3.tick_params(labelsize=10)
ax3.text(text_x_pos, text_y_pos, '$(d)$', transform=ax3.transAxes, fontsize=10)

# Second complemtarity variable μ
ax4 = plt.subplot(235)
ax4.set_prop_cycle('color', colour_split)
#ax3.plot(t, μ, linewidth=1)
ax4.plot(t, μ_analytic, linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$\mu$ (N/mm)', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(0, np.max(μ_analytic))
ax4.minorticks_on()
ax4.tick_params(labelsize=10)
ax4.text(text_x_pos, text_y_pos, '$(e)$', transform=ax4.transAxes, fontsize=10)

# First complemtarity variable λ
ax5 = plt.subplot(236)
ax5.set_prop_cycle('color', colour_split)
#ax2.plot(t, λ, linewidth=1)
ax5.plot(t, λ_analytic, linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$\lambda$ (ms$^{-1}$)', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(0, 0.6)
ax5.minorticks_on()
ax5.tick_params(labelsize=10)
ax5.text(text_x_pos, text_y_pos, '$(f)$', transform=ax5.transAxes, fontsize=10)

# Tighten the layout
plt.tight_layout()

plt.savefig(save_folder + "Figure_2.pdf")

plt.show()
