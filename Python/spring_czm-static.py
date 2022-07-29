import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import siconos.numerics as sn
import siconos

# Declare the name of the folder to save the figure
save_folder = "/path/to/your/folder/goes/here/"

# Material constants
# Critical traction
σ_c = 0.5
# Critical opening length
δ_c = 1
# Stiffness of the bar. For ill-posed simulation, replace with 0.5
E = 5.0  # 0.5
# Length of the bar
l = 1
# Cross sectional area of the bar (and CZM)
S_val = 1

# Number of time-steps
n = 4000
# Maximum integration time
tmax = 8.0

# Set the number of nodes to two, and construct the vectors and stiffness matrix
u = np.zeros((2, 1))
F = np.zeros((2, 1))
K = np.array([[E*S_val/l, -E*S_val/l], [-E*S_val/l, E*S_val/l]])

# Specify the control vector
control = np.zeros((2), dtype=bool)
control[1] = True
# Specify the tributary matrix
S = [[S_val]]
# Specify the selection matrix and its transpose
H = np.array([[1, 0]])
H_T = np.transpose(H)
b = [0.0]

# We can set the Siconos LCP solver here. Lemke is a direct method using pivoting,
# PGS (Projective Gauss-Seidel) is an iterative solver, ENUM enumerates all of the
# solutions (in most general case LCP_size^2)
id = sn.SICONOS_LCP_LEMKE
#id = sn.SICONOS_LCP_PGS
#id = sn.SICONOS_LCP_ENUM

# Initial values of intactness and λ and μ complemtarity variables, and thermodynamic force "A"
β = [1.]
λ = [0.]
μ = [0.]
A = [0.]

# Set the initial value of the normal jump displacement in the cohesive zone
u_n = [0.]

# Define a loading protocol (load-unload-reload)
def load_0(t):
    if t <= 1.0:
        u = 0.5*t
    elif 1.0 < t <= 3.0:
        u = 1 - 0.5*t
    elif t > 3.0:
        u  = - 2 + 0.5*t
    return u


# Define the load
load = load_0

# Create the time steps and print them
t = np.arange(0.0, tmax + tmax/n, tmax/n)
print(t)

# Initialise the driving displacement as the first value of the load
U = [load(0)]

# Initialise the reaction force and slack variable ν
r_N = [0.0]
ν = [r_N[-1] + β[-1]*σ_c]


# Define a function which inputs the stiffness matrix, the force, the displacement
# and the control vector, to enforce the boundary conditions
def boundary_condition_enforcement(K, F, u, control):
    F_bar = F.copy()
    # Loop through the controlled displacement
    for i in range(np.shape(u)[0]):
        # If it's a controlled velocity, calculate the impulse from the augmented
        # mass matrix.
        if control[i]:
            F_bar[i] = K[i, i]*u[i]
            # Then subtract its effect from all the other entries
            for j in range(np.shape(K)[0]):
                if j != i:
                    # Make sure we don't interfere with any of the other controlled
                    # velocities
                    if not control[j]:
                        F_bar[j] = F_bar[j] - K[j, i]*u[i]
    return F_bar


# Define a function that modifies the stiffness matrix to enforce the boundary
# conditions (we will only need to call this once, hence why it's a seperate function)
def modified_stiffness_matrix(K, control):
    K_bar = K.copy()
    # Loop through the controlled velocities
    for i in range(np.shape(control)[0]):
        # If it's a controlled velocity, change the matrx
        if control[i]:
            # Reset the augmented masss matrix
            K_bar[i, :] = 0.0
            K_bar[:, i] = 0.0
            K_bar[i, i] = K[i, i]
    return K_bar


def test(u, F, K, β, A, u_n, μ, λ, ν, U, r_N, control):
    # Let it know that it's an LCP of size 3
    lcp_size = 3
    # Assemble the modified stiffness matrix once
    K_bar = modified_stiffness_matrix(K, control)
    # Invert it
    K_bar_inv = np.linalg.inv(K_bar)
    # Calculate the shared term
    shared_term = np.matmul(H, np.matmul(K_bar_inv, np.matmul(H_T, S)))
    # Assemble the L matrix of the LCP
    L = np.block([[0.0, -1.0, 0.0],
         [1.0, σ_c*δ_c - (σ_c**2)*shared_term, -σ_c*shared_term],
         [0, σ_c*shared_term, shared_term]])

    # Now loop through each time step
    for k in range(n + 1):
        # Print the step
        print('step #', k)
        # Append the output of the load function to the driving displacement, and print it
        U.append(load(t[k]))
        print('U', U[-1])

        # Declare some holding vectors
        u_vec = u[:, -1]
        u_vec[1] = U[-1]
        F_vec = F[:, -1]

        # Enforce the boundary conditions
        F_bar = boundary_condition_enforcement(K, F_vec, u_vec, control)
        F_bar = np.reshape(F_bar, (2, 1))

        # Now give the vector part of the LCP
        q = np.block([np.array([[β[-1]]]),
                      np.block([-σ_c*(δ_c*(β[-1] - 1.0) + np.matmul(H, np.matmul(K_bar_inv, (F_bar - σ_c*np.matmul(H_T, S)*β[-1]))) + b[0])]),
                      np.block([np.matmul(H, np.matmul(K_bar_inv, (F_bar - σ_c*np.matmul(H_T, S)*β[-1]))) + b[0]])])
        # Try changing q to a single dimension (for the sake of Siconos)
        q = np.reshape(q, (3))
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
        # Check that it successfully completes
        assert(not info)
        # Print the solutions
        print(w, z)

        # Append the value of β with the the most recent value of β, w[0]
        β.append(w[0])
        # Append the value of A with the most recent value of A, w[1]
        A.append(w[1])
        # Append the value of u_n with the most recent value of ν, w[2]
        u_n.append(w[2])
        # Append the value of μ with the most recent value of μ, z[0]
        μ.append(z[0])
        # Append the value of λ with the most recent value of λ, z[1] divided by the time-step
        λval = z[1]/(tmax/n)
        λ.append(λval)
        # Append with latest value of ν
        ν.append(z[2])

        # Append the reaction force value
        r_N.append(ν[-1] - β[-1]*σ_c)

        # Calculate the displacements
        u_new = np.zeros((2, 1))
        u_new[:, 0] = [u_n[-1], U[-1]]
        u_calc = np.matmul(K_bar_inv, (F_bar + np.matmul(H_T, np.matmul(S, [[r_N[-1]]]))))
        # Add u and F vals
        u = np.concatenate((u, u_calc), axis=1)
        F = np.concatenate((F, np.zeros((2, 1))), axis=1)

        # Print the value of β
        print('β = ', β[-1])
    return


# Run test
test(u, F, K, β, A, u_n, μ, λ, ν, U, r_N, control)

# Calculate the cohesive force
r_c = σ_c*np.array(β)

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
line_split = ['-', '--', '-.', ':', '-', '--']

# Normal displacement with time
ax0 = plt.subplot(231)
ax0.set_prop_cycle('color', colour_split)
ax0.plot(t, u_n, linestyle=line_split[0], linewidth=1, label=r'$u_{1}$')
ax0.plot(t, U, linestyle=line_split[1], linewidth=1, label=r'$u_{2}$')
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$u$ (mm)', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(np.min(U), np.max(U))
ax0.minorticks_on()
ax0.tick_params(labelsize=10)
text_x_pos = 0.015
text_y_pos = 0.92
ax0.text(text_x_pos, text_y_pos, '$(a)$', transform=ax0.transAxes, fontsize=10)

# Intactness with time
ax1 = plt.subplot(232)
ax1.set_prop_cycle('color', colour_split)
ax1.plot(t, β, linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$\beta$', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(0, 1.15)
ax1.minorticks_on()
ax1.tick_params(labelsize=10)
ax1.text(text_x_pos, text_y_pos, '$(b)$', transform=ax1.transAxes, fontsize=10)

# Intactness with displacement
ax2 = plt.subplot(233)
ax2.set_prop_cycle('color', colour_split)
ax2.plot(u_n, β, linewidth=1)
plt.xlabel(r'$u_{\hbox{\tiny{N}}}$ (mm)', fontsize=10)
plt.ylabel(r'$\beta$', fontsize=10)
plt.xlim(np.min(u_n), np.max(u_n))
plt.ylim(0, 1.1)
ax2.minorticks_on()
ax2.tick_params(labelsize=10)
ax2.text(text_x_pos, text_y_pos, '$(c)$', transform=ax2.transAxes, fontsize=10)

# Driving force with time
ax3 = plt.subplot(234)
ax3.set_prop_cycle('color', colour_split)
ax3.plot(t, A, linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$A^{\sf r}$ (N/mm)', fontsize=10)
plt.xlim(np.min(t), np.max(t))
plt.ylim(0, 0.25)
ax3.minorticks_on()
ax3.tick_params(labelsize=10)
ax3.text(text_x_pos, text_y_pos, '$(d)$', transform=ax3.transAxes, fontsize=10)

# Reaction and cohesive forces with time
ax4 = plt.subplot(235)
ax4.plot(t, r_N, color=colour_split[2], linestyle=line_split[2], linewidth=1, label = r'$r_{\hbox{\tiny{N}}}$')
ax4.plot(t, r_c, color=colour_split[3], linestyle=line_split[3], linewidth=1, label = r'$r_{\hbox{\tiny{N}}}^{\sf c}$')
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$r_{\hbox{\tiny{N}}}^{(\sf c)}$ (N/mm$^{2}$)', fontsize=10)
plt.xlim(np.min(t), np.max(t))
plt.ylim(np.min(r_N), np.max(r_N))
ax4.minorticks_on()
ax4.tick_params(labelsize=10)
ax4.text(text_x_pos, text_y_pos, '$(e)$', transform=ax4.transAxes, fontsize=10)

# λ with t
ax5 = plt.subplot(236)
ax5.set_prop_cycle('color', colour_split)
ax5.plot(t, λ, linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$\lambda$ (ms$^{-1}$)', fontsize=10)
plt.xlim(np.min(t), np.max(t))
plt.ylim(np.min(λ), 1.2*np.max(λ))
ax5.minorticks_on()
ax5.tick_params(labelsize=10)
ax5.text(text_x_pos, text_y_pos, '$(f)$', transform=ax5.transAxes, fontsize=10)

# Tighten the layout
plt.tight_layout()

# Create the space and make the legend
fig.subplots_adjust(top=0.92)
fig.legend(loc=9, ncol=4, borderaxespad=0.)

# Save the well-posed figure
plt.savefig(save_folder + "Figure_4.pdf")
# Alternatively, save the ill-posed figure
#plt.savefig(save_folder + "Figure_5.pdf")

plt.show()
