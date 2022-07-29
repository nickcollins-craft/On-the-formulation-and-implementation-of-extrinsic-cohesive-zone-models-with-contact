import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import siconos.numerics as sn
import matplotlib.pyplot as plt

# Declare the name of the folder to save the figure
save_folder = "/path/to/your/folder/goes/here/"

# Material constants
# Critical traction (MPa)
σ_c = 1.0
# Critical opening length (mm)
δ_c = 1.0
# Mass matrix (g)
M = np.array([[2.5, 0.0], [0.0, 2.5]])
# Newton's coefficient of restitution
e = 0.0
# Stiffness of the bar (MPa)
E = 10
# Length of the bar (mm)
l_bar = 1
# Area matrix (mm^2)
S = np.array([[1.0]])
# Stiffness matrix (MPa)
K = (E*S[0, 0]/l_bar)*np.array([[1.0, -1.0], [-1.0, 1.0]])

# Set the time steps and maximum time (ms)
n_time_steps = 4000
tmax = 8.0
# Set the θ parameter
θ = 1.0

# Define a loading function in force
def load_1(t):
    f = (1.5*σ_c)*np.exp(0.25*t)*np.sin(np.pi*t)
    return f


# Define the load (in force)
load = load_1
F = [load(0)]

# We can set the Siconos LCP solver here. Lemke is a direct method using pivoting,
# PGS (Projective Gauss-Seidel) is an iterative solver
id = sn.SICONOS_LCP_LEMKE
# id = sn.SICONOS_LCP_PGS

# Initial values of intactness and λ and μ complemtarity variables, thermodynamic
# force "A", normal displacement and velocity, and percussion
β = [1.]
λ = [0.]
μ = [0.]
A = [0.]
u_n = [0.]
v_n = [0.]
p_n = [0.]

# Reshape for easy multiplication
β = np.reshape(np.array(β, ndmin=2), (np.max(np.shape(β)), 1))
A = np.reshape(np.array(A, ndmin=2), (np.max(np.shape(A)), 1))
v_n = np.reshape(np.array(v_n, ndmin=2), (np.max(np.shape(v_n)), 1))
λ = np.reshape(np.array(λ, ndmin=2), (np.max(np.shape(λ)), 1))
μ = np.reshape(np.array(μ, ndmin=2), (np.max(np.shape(μ)), 1))
p_n = np.reshape(np.array(p_n, ndmin=2), (np.max(np.shape(p_n)), 1))
u_n = np.reshape(np.array(u_n, ndmin=2), (np.max(np.shape(u_n)), 1))

# Selection matrix H and translation term b
H = np.array([1.0, 0.0], ndmin=2)
H_T = np.transpose(H)
b = np.array([0.0])

# Tolerance condition
R = 1e-8

# Define the time vector
h = tmax/n_time_steps
t = np.arange(0.0, tmax + h, h)

# Initialise the displacements and velocities
u = np.array([[0.0], [0.0]], ndmin=2)
v = np.array([[0.0], [0.0]], ndmin=2)

def test_force(F, u, v, β, A, v_n, μ, λ, p_n, u_n):
    # Calculate the augmented mass matrix M_hat
    M_hat = M + ((h*θ)**2)*K
    M_hat_inv = np.linalg.inv(M_hat)
    # Make the matrix multiplication
    matrices = np.matmul(H, np.matmul(M_hat_inv, np.matmul(H_T, S)))

    # Tell how many CZMs
    czm_count = 1

    # Create some vectors so that numpy concatenation behaves properly
    new_β = np.zeros((czm_count, 1))
    new_A = np.zeros_like(new_β)
    new_v_n = np.zeros_like(new_β)
    new_μ = np.zeros_like(new_β)
    new_λ = np.zeros_like(new_β)
    new_p_n = np.zeros_like(new_β)
    new_u_n = np.zeros_like(new_β)

    # Loop through the time steps
    for k in range(n_time_steps):

        # Print the step
        print("Time step #", k)
        # Append the loading force
        F.append(load(t[k]))
        print("F = ", F[-1])
        F_new = np.array([[0.0], [F[-1]]], ndmin=2)
        if k == 0:
            F_vec = F_new
        else:
            F_vec = np.concatenate((F_vec, F_new), axis=1)

        # Calculate the free-flight impulse
        if k > 0:
            i_hat = np.matmul(M - (h**2)*θ*(1 - θ)*K, v[:, -1]) - h*np.matmul(K, u[:, -1]) + h*(θ*F_vec[:, -1] + (1 - θ)*F_vec[:, -2])
        else:
            i_hat = np.matmul(M - (h**2)*θ*(1 - θ)*K, v[:, -1]) - h*np.matmul(K, u[:, -1]) + h*θ*F_vec[:, -1]

        # Calculate a shared term that goes on multiple lines of the LCP
        shared_term = np.matmul(H, np.matmul(M_hat_inv, (i_hat + np.matmul(H_T, np.matmul(S, -h*σ_c*(β[:, -1]))))))

        # Now apply the test for whether there is contact or not
        if u_n[:, -1] + (h/2)*v_n[:, -1] > R:
            print("Free fall")
            # No contact case.
            czm_variables = 2
            # Now we can set up the LCP system
            L = np.block([[0*np.eye(czm_count), -1*np.eye(czm_count)],
                          [1*np.eye(czm_count), σ_c*δ_c*θ*np.eye(czm_count) - (h**2)*(θ**3)*(σ_c**2)*matrices]])
            q = np.concatenate((np.transpose(np.array(β[:, -1], ndmin=2)),
                                np.transpose(np.array(-σ_c*δ_c*(β[:, -1] - h*(1 - θ)*λ[:, -1] - 1) - σ_c*(u_n[:, -1] + h*((1 - θ)*v_n[:, -1] + θ*shared_term)), ndmin=2))))
            # Tell Siconos how to understand the problem (i.e. that it's an LCP)
            lcp = sn.LCP(L, q)

            # Declare two zero vectors that will hold the solutions of the LCP. These
            # are vectors that solve the system Lz + q = w, where z and w are normal.
            z = np.zeros((czm_variables,), np.float64)
            w = np.zeros_like(z)
            # Extract the solver options from Siconos given the type of LCP algorithm
            options = sn.SolverOptions(id)
            # Run the linear complementarity solver, given the LCP, the solution vectors and the options
            info = sn.linearComplementarity_driver(lcp, z, w, options)
            # Check that it completes properly
            assert not(info)
            # Print the iterations and the solution residuals
            print('iter = ', options.iparam[sn.SICONOS_IPARAM_ITER_DONE])
            print('error = ', options.dparam[sn.SICONOS_DPARAM_RESIDU])
            # Print the solutions
            print(w, z)

            # Append the new values
            new_β[:, 0] = w[0:czm_count]
            β = np.concatenate((β, new_β), axis=1)
            new_A[:, 0] = w[czm_count:2*czm_count]
            A = np.concatenate((A, new_A), axis=1)
            new_μ[:, 0] = z[0:czm_count]
            μ = np.concatenate((μ, new_μ), axis=1)
            new_λ[:, 0] = z[czm_count:2*czm_count]/h
            λ = np.concatenate((λ, new_λ), axis=1)
            # Append the value of p_n with 0
            new_p_n[:, 0] = np.zeros((czm_count))
            p_n = np.concatenate((p_n, new_p_n), axis=1)

            # Make the updates to the rest of the system
            cohesive_impulses = -h*σ_c*((1 - θ)*β[:, -2] + θ*β[:, -1]) + p_n[:, -1]
            v_update = np.array(np.matmul(M_hat_inv, (i_hat + np.matmul(H_T, np.matmul(S, cohesive_impulses)))), ndmin=2)
            v_update = np.transpose(v_update)
            u_update = np.array(u[:, -1] + h*((1 - θ)*v[:, -1] + θ*v_update[:, -1]), ndmin=2)
            u_update = np.transpose(u_update)
            v = np.append(v, v_update, axis=1)
            u = np.append(u, u_update, axis=1)
            # Insert relevant values into u_n and v_n
            new_v_n = np.matmul(H, v_update)
            new_u_n = np.matmul(H, u_update) + b
            v_n = np.append(v_n, new_v_n, axis=1)
            u_n = np.concatenate((u_n, new_u_n), axis=1)
        else:
            print("Contact step")
            # Contact case.
            czm_variables = 3

            # Now we can set up the LCP system
            L = np.block([[0*np.eye(czm_count), -1*np.eye(czm_count), 0*np.eye(czm_count)],
                          [1*np.eye(czm_count), σ_c*δ_c*θ*np.eye(czm_count) - (h**2)*(θ**3)*(σ_c**2)*matrices, -h*θ*σ_c*matrices],
                          [0*np.eye(czm_count), h*(θ**2)*σ_c*matrices, matrices]])
            q = np.concatenate((np.transpose(np.array(β[:, -1], ndmin=2)),
                                np.transpose(np.array(-σ_c*δ_c*(β[:, -1] - h*(1 - θ)*λ[:, -1] - 1) - σ_c*(u_n[:, -1] + h*((1 - θ)*v_n[:, -1] + θ*shared_term)), ndmin=2)),
                                np.transpose(np.array(shared_term + e*v_n[:, -1], ndmin=2))))
            # Tell Siconos how to understand the problem (i.e. that it's an LCP)
            lcp = sn.LCP(L, q)

            # Declare two zero vectors that will hold the solutions of the LCP. These
            # are vectors that solve the system Lz + q = w, where z and w are normal.
            z = np.zeros((czm_variables,), np.float64)
            w = np.zeros_like(z)
            # Extract the solver options from Siconos given the type of LCP algorithm
            options = sn.SolverOptions(id)
            # Run the linear complementarity solver, given the LCP, the solution vectors and the options
            info = sn.linearComplementarity_driver(lcp, z, w, options)
            # Check that it completes properly
            assert not(info)
            # Print the iterations and the solution residuals
            print('iter = ', options.iparam[sn.SICONOS_IPARAM_ITER_DONE])
            print('error = ', options.dparam[sn.SICONOS_DPARAM_RESIDU])
            # Print the solutions
            print(w, z)

            # Append the new values
            new_β[:, 0] = w[0:czm_count]
            β = np.concatenate((β, new_β), axis=1)
            new_A[:, 0] = w[czm_count:2*czm_count]
            A = np.concatenate((A, new_A), axis=1)
            new_v_n[:, 0] = w[2*czm_count:czm_count*czm_variables] - e*v_n[:, -1]
            v_n = np.concatenate((v_n, new_v_n), axis=1)
            new_μ[:, 0] = z[0:czm_count]
            μ = np.concatenate((μ, new_μ), axis=1)
            new_λ[:, 0] = z[czm_count:2*czm_count]/h
            λ = np.concatenate((λ, new_λ), axis=1)
            new_p_n[:, 0] = z[2*czm_count:czm_count*czm_variables]
            p_n = np.concatenate((p_n, new_p_n), axis=1)

            # Make the updates to the rest of the system
            cohesive_impulses = -h*σ_c*((1 - θ)*β[:, -2] + θ*β[:, -1]) + p_n[:, -1]
            v_update = np.array(np.matmul(M_hat_inv, (i_hat + np.matmul(H_T, np.matmul(S, cohesive_impulses)))), ndmin=2)
            v_update = np.transpose(v_update)
            u_update = np.array(u[:, -1] + h*((1 - θ)*v[:, -1] + θ*v_update[:, -1]), ndmin=2)
            u_update = np.transpose(u_update)
            v = np.append(v, v_update, axis=1)
            u = np.append(u, u_update, axis=1)
            # Insert relevant values into u_n
            new_u_n = np.matmul(H, u_update) + b
            u_n = np.concatenate((u_n, new_u_n), axis=1)

        print("β = ", β[:, -1])
    return F, u, v, β, A, v_n, μ, λ, p_n, u_n


results = test_force(F, u, v, β, A, v_n, μ, λ, p_n, u_n)
F = results[0]
u = results[1]
v = results[2]
β = results[3]
A = results[4]
v_n = results[5]
μ = results[6]
λ = results[7]
p_n = results[8]
u_n = results[9]

# Time to plot!
# Set the width in inches of the desired plot (everything in the brackets is
# basically full width of an A4 page with 1.5 cm margins)
width_in_inches = (8.27-2*1.5/2.54)/1
# Create the figure with the requisite number of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(width_in_inches, 0.625*width_in_inches))
dof_index = [0, 1]
colour_split = plt.cm.viridis(np.linspace(0, 1, 5))
line_split = ['-', '--', '-.', ':']
# TeX the written elements so that it looks good (comment out until final run
# because calling TeX is *slow*)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Normal displacement with time
ax0 = plt.subplot(231)
ax0.set_prop_cycle('color', colour_split)
for index in dof_index:
    ax0.plot(t[:-1], u[index, :], linestyle=line_split[index], linewidth=1, label=r'$u_{}$'.format(index + 1))
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$u$ (mm)', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(np.min(u), np.max(u))
ax0.minorticks_on()
ax0.tick_params(labelsize=10)
text_x_pos = 0.015
text_y_pos = 0.92
ax0.text(text_x_pos, text_y_pos, '$(a)$', transform=ax0.transAxes, fontsize=10)

# Velocity with time
ax1 = plt.subplot(232)
ax1.set_prop_cycle('color', colour_split)
for index in dof_index:
    ax1.plot(t[:-1], v[index, :], linestyle=line_split[index], linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$v$ (mm/ms)', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(1.05*np.min(v), 1.05*np.max(v))
ax1.minorticks_on()
ax1.tick_params(labelsize=10)
ax1.text(text_x_pos, text_y_pos, '$(b)$', transform=ax1.transAxes, fontsize=10)

# β with time
ax2 = plt.subplot(233)
ax2.set_prop_cycle('color', colour_split)
ax2.plot(t[:-1], β[0, :], linestyle=line_split[0], linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$\beta$', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(0, 1.2)
ax2.minorticks_on()
ax2.tick_params(labelsize=10)
ax2.text(text_x_pos, text_y_pos, '$(c)$', transform=ax2.transAxes, fontsize=10)

# A with time
ax3 = plt.subplot(234)
ax3.set_prop_cycle('color', colour_split)
ax3.plot(t[:-1], A[0, :], linestyle=line_split[0], linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$A^{\sf r}$ (N/mm)', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(0.0, 0.6)
ax3.minorticks_on()
ax3.tick_params(labelsize=10)
ax3.text(text_x_pos, text_y_pos, '$(d)$', transform=ax3.transAxes, fontsize=10)

# Percussion with time
ax4 = plt.subplot(235)
ax4.set_prop_cycle('color', colour_split)
ax4.plot(t[:-1], p_n[0, :], linestyle=line_split[0], linewidth=1)
plt.xlabel(r'$t$ (ms)', fontsize=10)
plt.ylabel(r'$p_{\hbox{\tiny{N}}}$ (N$\cdot$ms/mm$^{2}$)', fontsize=10)
plt.xlim(0, tmax)
plt.ylim(0, 1.05*np.max(p_n))
ax4.minorticks_on()
ax4.tick_params(labelsize=10)
ax4.text(text_x_pos, text_y_pos, '$(e)$', transform=ax4.transAxes, fontsize=10)

# Intactness with displacement
ax5 = plt.subplot(236)
ax5.set_prop_cycle('color', colour_split)
ax5.plot(u_n[0, :], β[0, :], linestyle=line_split[0], linewidth=1)
plt.xlabel(r'$u_{\hbox{\tiny{N}}}$ (mm)', fontsize=10)
plt.ylabel(r'$\beta$', fontsize=10)
plt.xlim(0, np.max(u_n))
plt.ylim(0, 1.1)
ax5.minorticks_on()
ax5.tick_params(labelsize=10)
ax5.text(text_x_pos, text_y_pos, '$(f)$', transform=ax5.transAxes, fontsize=10)

# Tighten the layout
plt.tight_layout()

# Create the space and make the legend
fig.subplots_adjust(top=0.93)
fig.legend(loc=9, ncol=2, borderaxespad=0.)

plt.savefig(save_folder + "Figure_7.pdf")

plt.show()
