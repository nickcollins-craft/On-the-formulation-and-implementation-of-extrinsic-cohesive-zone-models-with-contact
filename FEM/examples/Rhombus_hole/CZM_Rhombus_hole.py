import sys
import os
import siconos.numerics as sn
import numpy as np
import siconos
sys.path.append('../../src/')
from fem import *
from mesh import *
import matplotlib.pyplot as plt

# Declare the name of the folder to save the output
save_folder = "/path/to/your/folder/goes/here/"

# Choose the mesh we operate on
mesh_angle = '90'

# Load the mesh (we assume the mesh is in the working folder)
filename = 'Rhombus_' + mesh_angle + '_degree'
Rhombus_hole_mesh = gmsh_mesh(filename)
mesh = Rhombus_hole_mesh._mesh
physical_name = Rhombus_hole_mesh._physical_name

# Set the maximum time, the number of time-steps and the loading rate
# number of time-steps
n_time_steps = 400
# Maximum integration time (in ms)
tmax = 1600
# Set the θ parameter
θ = 0.5
# Set the displacement target
d_target = 1.6
# Displacement rate
d_rate = d_target/tmax

# Create a material, using the parameters from Doitrand et al. (2019), for PMMA.
# We will use units in N, mm, g, MPa etc
E = 1600
ν = 0.37
ρ = 1.18E-3
material = material(E, ν, ρ, load_type='plane_strain')

# Write the maximum time-step once cracking occurs
h_crack = 1e-4

# Create the CZM material
# Critical traction (MPa), using value for average stress criterion
σ_c = 80
# Griffith fracture energy in N/mm
Gc = 0.25
# Critical opening length (mm), from Gc
δ_c = 2*Gc/σ_c
# Newton's coefficient of restitution
e = 0.0

# Tolerance condition
R = 1e-8

# We can set the Siconos LCP solver here. Lemke is a direct method using pivoting
id = sn.SICONOS_LCP_LEMKE

# Create the fem model and compute the stiffness and mass matrices
model = fem_model(Rhombus_hole_mesh._mesh, thickness=10.0, mass_type='consistent', integration_type='four_point')
model.compute_stiffness_matrix(material)
model.compute_mass_matrix(material)
K = model.K
M = model.M
# Set the forces
F = np.zeros(model._n_dof)

# Define the displacement controlled load in terms of the velocity *only*, noting
# that it's negative because we compress
def displacement_load(t):
    v = -d_rate
    return v


# Determine the applied punctual force nodes. The convention is punctual force
# load is applied on cells of type vertex
e_cnt = 0
punctual_force_nodes = []
for fd in physical_name:
    if fd == 'Applied displacement':
        punctual_force_nodes.extend(physical_name[fd]['nodes'])


# Determine the boundary conditions.
dirichlet_dof = []
contact_dof = []
punctual_force_dof = []
dirichlet_contact_dof = []
# Initialise the values of the CZM variables
β = []
A = []
v_n = []
λ = []
μ = []
p_n = []
# Initialise the energetic variables we also want to track
Ψ_ε = []
Ψ_s = []
Ψ_K = []
work = []
contact_work = []
# And some auxiliary variables we want to track too
u_n = []
y_pos = []
d_c_y_pos = []
for fd in physical_name:
    if 'Dirichlet BC' in fd:
        for n in physical_name[fd]['nodes']:
            # Dirichlet condition applied only in the y-direction, for symmetry BC
            dirichlet_dof.extend([2*n + 1])
    if 'Contact BC' in fd:
        for n in physical_name[fd]['nodes']:
            # Check whether the point is in both the contact boundary condition
            # and the Dirichlet boundary condition. If so, we will not count it
            # in the list of czm nodes (for numerical reasons)
            Dirichlet_Contact = False
            for dirichlet_index in dirichlet_dof:
                if dirichlet_index == 2*n or dirichlet_index == 2*n + 1:
                    Dirichlet_Contact = True
                    dirichlet_contact_dof.extend([2*n])
                    d_c_y_pos.extend([mesh.points[n][1]])
            if Dirichlet_Contact:
                print("Dirichlet - contact point found at node #", n)
                continue
            else:
                # Enter *only* the normal opening in the list of contact dofs
                # (it's in the x-direction this time)
                contact_dof.extend([2*n])
                # Insert the value of the CZM variables to the end of the list
                β.extend([1.])
                A.extend([0.])
                v_n.extend([0.])
                λ.extend([0.])
                μ.extend([0.])
                p_n.extend([0.])
                u_n.extend([0.])
                y_pos.extend([mesh.points[n][1]])
                print("y_pos = ", y_pos[-1])
    if 'Applied displacement' in fd:
        for n in physical_name[fd]['nodes']:
            # Applied in both directions
            punctual_force_dof.extend([2*n, 2*n + 1])

# Calculate a matrix giving the distances of the contact points from each other
distance_matrix = np.zeros((len(contact_dof), len(contact_dof)))
for node_index_1 in range(np.shape(distance_matrix)[0]):
    for node_index_2 in range(np.shape(distance_matrix)[1]):
        distance_matrix[node_index_1, node_index_2] = np.abs(y_pos[node_index_1] - y_pos[node_index_2])

# Now work out the tributary areas for each czm node, starting by combining them
# with the dirichlet_contact_dofs
full_y_pos = np.concatenate((y_pos, d_c_y_pos))
# Create the vector that will store the areas
area_vector = np.zeros((len(contact_dof)))
# Iterate through the czm nodes and calculate the distances of the nodes to the
# left and right
for node_index in range(len(contact_dof)):
    # Get the case for the bottom czm boundary, with no czm node below
    if y_pos[node_index] == np.min(y_pos):
        # Add half the area above, from the distance to the next node to the
        # right, once again ignoring the self-distance
        area_vector[node_index] = area_vector[node_index] + 0.5*model._thickness*np.sort(distance_matrix[node_index, :])[1]
    # Now get the case for the upper czm boundary
    elif y_pos[node_index] == np.max(y_pos):
        # Now get the case where it's the top-most czm node, but there is a
        # czm-Dirichlet node that we have removed from the problem
        if full_y_pos[node_index] != np.max(full_y_pos):
            # Add all of the area between the czm-Dirichlet node and the true
            # czm node
            area_vector[node_index] = area_vector[node_index] + model._thickness*abs(np.max(full_y_pos) - y_pos[node_index])
            # Add half the area below, from the y position that is ranked one
            # higher. Take the second index because the first is the "self-distance"
            area_vector[node_index] = area_vector[node_index] + 0.5*model._thickness*np.sort(distance_matrix[node_index, :])[1]
        # Now get the case where there is no czm-Dirichlet node
        else:
            # Add half the area to the below, from the distance to the next node
            # below, taking the second entry to ignore the self-distance
            area_vector[node_index] = area_vector[node_index] + 0.5*model._thickness*np.sort(distance_matrix[node_index, :])[1]
    # Now solve all the other nodes
    else:
        # Add half the area above and below the nodes, taking the second and third
        # entries to ignore the self-distance
        area_vector[node_index] = area_vector[node_index] + 0.5*model._thickness*np.sort(distance_matrix[node_index, :])[1]
        area_vector[node_index] = area_vector[node_index] + 0.5*model._thickness*np.sort(distance_matrix[node_index, :])[2]

# Now convert the area vector into the S matrix
S = np.eye(len(contact_dof))
for i in range(len(contact_dof)):
    S[i, i] = area_vector[i]

# Turn the CZM variables into numpy arrays of dimension 2 (for numpy reasons), and
# enforce the orientation that we want
β = np.reshape(np.array(β, ndmin=2), (np.max(np.shape(β)), 1))
A = np.reshape(np.array(A, ndmin=2), (np.max(np.shape(A)), 1))
v_n = np.reshape(np.array(v_n, ndmin=2), (np.max(np.shape(v_n)), 1))
λ = np.reshape(np.array(λ, ndmin=2), (np.max(np.shape(λ)), 1))
μ = np.reshape(np.array(μ, ndmin=2), (np.max(np.shape(μ)), 1))
p_n = np.reshape(np.array(p_n, ndmin=2), (np.max(np.shape(p_n)), 1))
u_n = np.reshape(np.array(u_n, ndmin=2), (np.max(np.shape(u_n)), 1))


# Define a function which inputs the augmented mass matrix, the free-flight impulse,
# the velocity and the control vector, to enforce the boundary conditions
def boundary_condition_enforcement(M_hat, i_hat, v, control):
    i_hat_bar = i_hat.copy()
    # Loop through the controlled velocities
    for i in range(np.shape(v)[0]):
        # If it's a controlled velocity, calculate the impulse from the augmented
        # mass matrix.
        if control[i]:
            i_hat_bar[i] = M_hat[i, i]*v[i]
            # Then subtract its effect from all the other entries
            for j in range(np.shape(M_hat)[0]):
                if j != i:
                    # Make sure we don't interfere with any of the other controlled
                    # velocities
                    if not control[j]:
                        i_hat_bar[j] = i_hat_bar[j] - M_hat[j, i]*v[i]
    return i_hat_bar


# Define a function that modifies the augmented mass matrix to enforce the boundary
# conditions (we will only need to call this once, hence why it's a seperate function)
def modified_augmented_mass_matrix(M_hat, control):
    M_hat_bar = M_hat.copy()
    # Loop through the controlled velocities
    for i in range(np.shape(control)[0]):
        # If it's a controlled velocity, change the matrx
        if control[i]:
            # Reset the augmented masss matrix
            M_hat_bar[i, :] = 0.0
            M_hat_bar[:, i] = 0.0
            M_hat_bar[i, i] = M_hat[i, i]
    return M_hat_bar


# Get all the points that lie on the contact boundary, where we put the CZM, and
# put them in the H matrix. Size by counting the number of dofs along the cohesive
# boundary and by the total number of dofs
H = np.zeros((len(contact_dof), model._n_dof))
# Now, we want to loop through, assigning 1s for czm #x at node #y
for [czm_dof, czm_index] in zip(contact_dof, range(len(contact_dof))):
    # We just want the normal opening mode, but we have already made this filtering
    # operation when we obtained the CZM dofs. So, having sorted all the czm dofs,
    # we make the czm_dof-th entry of the czm_index-th row 1. i.e. each row indicates
    # the position in the list of czms, each column indicates the position in the
    # list of all dofs
    H[czm_index, czm_dof] = 1
# Create the b vector as well, to finish the link between dofs and normal displacements
b = np.zeros((len(contact_dof), 1))
# Precalculate H^T
H_T = np.transpose(H)

# Create the vector of initial displacements and velocities (i.e. zeros)
u = np.zeros((model._n_dof, 1))
v = np.zeros((model._n_dof, 1))

# Start by creating an ordered list of all the known displacement dofs
k_dofs = np.sort(np.concatenate((dirichlet_dof, punctual_force_dof)))
# Create a mask
k_mask_vec = np.zeros((model._n_dof, 1), dtype=bool)
# Add the known dofs to the mask
k_mask_vec[k_dofs] = np.ones((len(k_dofs), 1), dtype=bool)

# Define the time steps
h_max = tmax/n_time_steps
t = 0.0


def test(F, u, v, β, A, v_n, μ, λ, p_n, u_n, K, M, k_mask_vec, punctual_force_dof, h_max, t, Ψ_ε, Ψ_s, Ψ_K, work, contact_work):
    # Say how many CZMs we solve
    czm_count = len(contact_dof)
    # Say how many variables are in the CZM
    czm_variables = 3

    # Create some vectors so that numpy concatenation behaves properly
    new_β = np.zeros((czm_count, 1))
    new_A = np.zeros_like(new_β)
    new_v_n = np.zeros_like(new_β)
    new_μ = np.zeros_like(new_β)
    new_λ = np.zeros_like(new_β)
    new_p_n = np.zeros_like(new_β)
    new_u_n = np.zeros_like(new_β)

    h = h_max
    t_vec = [t]

    # Get the augmented mass matrix
    M_hat = M + ((h*θ)**2)*K
    # Make the modifications that help to enforce the BCs
    M_hat_bar = modified_augmented_mass_matrix(M_hat, k_mask_vec)
    # Get the inversion too
    M_hat_bar_inv = np.linalg.inv(M_hat_bar)
    # Make the matrix multiplication. We keep what is written as US, VS, V^TS in
    # the note all together as "matrices", and we track the changing contacts
    # and cohesive zones by taking sub-vectors and matrices
    matrices = np.matmul(H, np.matmul(M_hat_bar_inv, np.matmul(H_T, S)))
    # But because W doesn't countain the surface area, we get it separately
    W = np.matmul(H, np.matmul(M_hat_bar_inv, H_T))

    # Temporary code to check the size of h, by setting h_max = h_crack, and then re-simulating
    #key_matrix = σ_c*δ_c*np.eye(czm_count) - (h**2)*(θ**2)*(σ_c**2)*matrices
    #eigenvalues = np.linalg.eigvals(key_matrix)
    #print("h = ", h)
    #print("min(λ) = ", np.min(eigenvalues))
    #input()

    # Declare a boolean vector for contact detection
    contact_vector = np.ones((czm_count), dtype=bool)
    # Declare a boolean vector for cohesion detection
    intact_vector = np.ones((czm_count), dtype=bool)
    # Declare an array that tells us whether individual variables are solved for,
    # and give it 2 dimensions for numpy reasons
    variable_vector = np.ones((1, czm_count*czm_variables), dtype=bool)

    # We can declare our L matrix, which doesn't change between time steps (most of the time)
    L = np.block([[0*np.eye(czm_count), -np.eye(czm_count), 0*np.eye(czm_count)],
                  [np.eye(czm_count), σ_c*(δ_c*np.eye(czm_count) - (h**2)*(θ**2)*σ_c*matrices), -h*θ*σ_c*matrices],
                  [0*np.eye(czm_count), h*θ*σ_c*matrices, matrices]])

    # Have a Boolean variable for whether we need to update L
    step_different = False
    # Have a variable that counts how many successful steps at size h
    successful_steps = 0
    # Count how many steps taken
    k = 0

    # Set a variable that tracks whether cracking has started
    is_cracked = False
    # Set a variable that counts how many time-steps since a node became fully cracked
    steps_since_cracking = 0
    # Set a variable that counts how many time-steps since a node in the FPZ stopped decohering
    steps_since_decohering = 0
    # Track size of inverse of intact_vector to track cracking
    inverse_intact_size = 0
    # Finally, a condition for crack arrest
    crack_arrested = False

    # For each time step
    while t < tmax + h and not(crack_arrested):
        # Print the step-size and time
        print("t = ", t)
        print("h = ", h)

        # If the time-step is different to the previous one, re-calculate the matrices
        # that depend on it
        if step_different:
            M_hat = M + ((h*θ)**2)*K
            M_hat_bar = modified_augmented_mass_matrix(M_hat, k_mask_vec)
            M_hat_bar_inv = np.linalg.inv(M_hat_bar)
            matrices = np.matmul(H, np.matmul(M_hat_bar_inv, np.matmul(H_T, S)))
            W = np.matmul(H, np.matmul(M_hat_bar_inv, H_T))
            L = np.block([[0*np.eye(czm_count), -np.eye(czm_count), 0*np.eye(czm_count)],
                          [np.eye(czm_count), σ_c*(δ_c*np.eye(czm_count) - (h**2)*(θ**2)*σ_c*matrices), -h*θ*σ_c*matrices],
                          [0*np.eye(czm_count), h*θ*σ_c*matrices, matrices]])

        # Create the force vector
        if t == 0.0:
            F = np.reshape(F, (model._n_dof, 1))
            F_vec = F

        # Only add the loading before cracking starts
        if not(is_cracked):
            # Enter the velocity boundary conditions, first the imposed
            v[punctual_force_dof, -1] = displacement_load(t)*np.ones((len(punctual_force_dof)))
            # Now go through and set the horizontal displacements to zero (no-slip)
            for n in punctual_force_dof:
                if n % 2 == 0:
                    v[n, -1] = 0.0
        else:
            v[punctual_force_dof, -1] = np.zeros((len(punctual_force_dof)))

        # Then the Dirichlet ones
        v[dirichlet_dof, -1] = np.zeros((len(dirichlet_dof)))

        # Calculate the free-flight impulse
        i_hat = np.matmul(M - (h**2)*θ*(1 - θ)*K, v[:, -1]) - h*np.matmul(K, u[:, -1])

        # Calculate the modified augmented mass matrix and free-flight impulse
        # which will enforce the boundary conditions
        i_hat_bar = boundary_condition_enforcement(M_hat, i_hat, v[:, -1], k_mask_vec)

        # Calculate the raw q vector for the LCP
        q = np.concatenate((np.transpose(np.array(β[:, -1], ndmin=2)),
                            np.transpose(np.array(-σ_c*(δ_c*(β[:, -1] - np.ones((len(β[:, -1])))) + u_n[:, -1] + h*(1 - θ)*np.matmul(H, v[:, -1]) + h*θ*np.matmul(H, np.matmul(M_hat_bar_inv, i_hat_bar)) - (h**2)*θ*σ_c*np.matmul(matrices, β[:, -1])), ndmin=2)),
                            np.transpose(np.array(np.matmul(H, np.matmul(M_hat_bar_inv, i_hat_bar)) - h*σ_c*np.matmul(matrices, β[:, -1]) + e*v_n[:, -1], ndmin=2))))

        # Cycle through the contact points to see whether they're in contact
        total_variables = czm_count*czm_variables
        contact_variables = czm_count
        for contact_index in range(np.shape(contact_vector)[0]):
            if u_n[contact_index, -1] + (h/2)*v_n[contact_index, -1] > R:
                # Not in contact, remove corresponding p_n from problem
                contact_vector[contact_index] = False
                variable_vector[0, czm_count*(czm_variables - 1) + contact_index] = False
                contact_variables = contact_variables - 1
                total_variables = total_variables - 1
            else:
                # Contact, retain corresponding p_n
                contact_vector[contact_index] = True
                variable_vector[0, czm_count*(czm_variables - 1) + contact_index] = True

        # Now cycle through the contact points to see whether they're fully decohered
        intact_variables = czm_count
        for contact_index in range(np.shape(intact_vector)[0]):
            if β[contact_index, -1] <= 1e-12:
                # Completely broken, say the node needs to be excluded
                intact_vector[contact_index] = False
                # Remove the β variable
                variable_vector[0, contact_index] = False
                # Then remove the SA variable
                variable_vector[0, czm_count + contact_index] = False
                # Count the number of intact nodes
                intact_variables = intact_variables - 1
                # Remove two variables from the total number, accounting for β and A
                total_variables = total_variables - 2

        # Create the mask to pick the submatrix
        mask = np.matmul(np.transpose(variable_vector), variable_vector)

        # Now we can set up the LCP system
        # Pick the submatrix and subvector that allow to us calculate only the relevant variables
        L_sub = np.reshape(L[mask], (total_variables, total_variables))
        q_sub = q[variable_vector[0, :], 0]

        # Tell Siconos how to understand the problem (i.e. that it's an LCP)
        lcp = sn.LCP(L_sub, q_sub)

        # Declare two zero vectors that will hold the solutions of the LCP. These
        # are vectors that solve the system Lz + q = w, where z and w are normal.
        z = np.zeros((total_variables,), np.float64)
        w = np.zeros_like(z)
        # Extract the solver options from Siconos given the type of LCP algorithm
        options = sn.SolverOptions(id)
        # Run the linear complementarity solver, given the LCP, the solution vectors and the options
        info = sn.linearComplementarity_driver(lcp, z, w, options)
        # Check that it completes properly, and if not halve the time-step
        if info:
            print("Time-step failed at h = ", h)
            h = h/2
            step_different = True
            successful_steps = 0
            continue
        # Otherwise keep it rolling
        else:
            # Get the easy variable updates
            # Calculate the βs calculated from the LCP
            new_β[intact_vector, 0] = w[0:intact_variables]
            # Calculate the βs for the fully broken interfaces
            new_β[np.invert(intact_vector), 0] = np.zeros((czm_count - intact_variables))
            β = np.concatenate((β, new_β), axis=1)
            # Calcualte the As from the LCP
            new_A[intact_vector, 0] = w[intact_variables:2*intact_variables]
            new_A[np.invert(intact_vector), 0] = np.zeros((czm_count - intact_variables))
            A = np.concatenate((A, new_A), axis=1)
            # Calculate the μs from the LCP
            new_μ[intact_vector, 0] = z[0:intact_variables]
            # Calculate the λs from the LCP
            new_λ[intact_vector, 0] = z[intact_variables:2*intact_variables]/h
            new_λ[np.invert(intact_vector), 0] = np.zeros((czm_count - intact_variables))
            λ = np.concatenate((λ, new_λ), axis=1)

            # Now get the more complicated cases depending on contact. We can fully
            # resolve p_n, but not v_n, where we need to solve the full velocities, then pick them out
            new_p_n[contact_vector, 0] = z[2*intact_variables:total_variables]
            new_p_n[np.invert(contact_vector), 0] = np.zeros((czm_count - contact_variables))
            p_n = np.concatenate((p_n, new_p_n), axis=1)

            # Make the updates to the rest of the system
            v_update = np.array(np.matmul(M_hat_bar_inv, (i_hat_bar + h*σ_c*np.matmul(H_T, np.matmul(S, θ*h*λ[:, -1] - β[:, -2])) + np.matmul(H_T, np.matmul(S, p_n[:, -1])))), ndmin=2)
            v_update = np.transpose(v_update)
            u_update = np.array(u[:, -1] + h*((1 - θ)*v[:, -1] + θ*v_update[:, -1]), ndmin=2)
            u_update = np.transpose(u_update)
            v = np.append(v, v_update, axis=1)
            u = np.append(u, u_update, axis=1)
            # Insert relevant values into u_n and v_n
            new_v_n = np.matmul(H, v_update)
            new_u_n = np.matmul(H, u_update) + b
            # Insert the values of v_n calculated from the LCP, as a back-up (in principle
            # the above should get them all correctly)
            new_v_n[contact_vector, 0] = w[2*intact_variables:total_variables] - e*v_n[contact_vector, -1]
            v_n = np.append(v_n, new_v_n, axis=1)
            u_n = np.concatenate((u_n, new_u_n), axis=1)

            # Calculate the μ for the fully broken interface by applying the equation
            # implemented in the LCP (but without decomposing into the LCP components)
            new_μ[np.invert(intact_vector), 0] = A[np.invert(intact_vector), -1] + σ_c*u_n[np.invert(intact_vector), -1] + σ_c*δ_c*(β[np.invert(intact_vector), -1] - np.ones((np.sum(np.invert(intact_vector)))))
            μ = np.concatenate((μ, new_μ), axis=1)

            # Print β so we can monitor
            print("β = ", β[:, -1])

            # Now write the energetic components
            Ψ_ε = np.append(Ψ_ε, 0.5*np.matmul(np.transpose(u[:, -1]), np.matmul(K, u[:, -1])))
            Ψ_s = np.append(Ψ_s, np.matmul(area_vector, σ_c*np.multiply(β[:, -1], u_n[:, -1]) + Gc*np.square(β[:, -1] - np.ones((len(β[:, -1]))))))
            Ψ_K = np.append(Ψ_K, 0.5*np.matmul(np.transpose(v[:, -1]), np.matmul(M, v[:, -1])))
            # Write the czm contributions
            contrib_czm = h*σ_c*np.matmul(H_T, np.matmul(S, ((1 - θ)*β[:, -2] + θ*β[:, -1]))) - np.matmul(H_T, np.matmul(S, p_n[:, -1]))
            # Back calculate the force F_k+θ
            F_bar = (1/h)*(np.matmul(M_hat, v[:, -1]) - i_hat + contrib_czm)
            F_bar = np.reshape(F_bar, (len(F_bar), 1))
            F_vec = np.concatenate((F_vec, F_bar), axis=1)
            # Then the power over the internval k, k+1
            P = np.matmul(F_vec[punctual_force_dof, -1], (1 - θ)*v[punctual_force_dof, -2] + θ*v[punctual_force_dof, -1])
            if t == 0.0:
                # Append to the work
                work = np.append(work, h*P)
                # Get the contact work
                contact_work_step = 0.5*np.matmul(v_n[:, -1], np.matmul(S, p_n[:, -1]))
            else:
                # Append to the work
                work = np.append(work, work[-1] + h*P)
                # Get the contact work
                contact_work_step = 0.5*np.matmul(v_n[:, -1] + v_n[:, -2], np.matmul(S, p_n[:, -1]))
            contact_work = np.append(contact_work, contact_work_step)

            # Prepare to write to vtk
            Rhombus_hole_mesh.vtk_prepare_output_displacement(u[:, -1])

            # Output strain for post-processing
            ε = model.compute_strain_at_gauss_points(u[:, -1])
            Rhombus_hole_mesh.vtk_prepare_output_strain(ε)

            # File writing
            file_write_folder = save_folder + "/Rhombus_hole/vtk/" + mesh_angle + "_degrees/"
            # Check that the folder exists and make it if not
            if not os.path.isdir(file_write_folder):
                os.mkdir(file_write_folder)
            foutput = '{0}{1:03d}.vtk'.format(filename, k)
            file_write = file_write_folder + foutput
            print(file_write)
            Rhombus_hole_mesh.vtk_finalize_output(file_write)

            # Check whether we have started cracking
            if not(is_cracked):
                if np.min(new_β) <= 1e-12:
                    is_cracked = True
                    h = h_crack
                    crack_step = True
            # Otherwise check if the crack has arrested
            else:
                # First check the fully cracked nodes
                cracked_nodes = np.sum(np.invert(intact_vector))
                if cracked_nodes - inverse_intact_size > 0:
                    inverse_intact_size = cracked_nodes
                    steps_since_cracking = 0
                else:
                    steps_since_cracking = steps_since_cracking + 1
                # Now check the decohering nodes
                if np.max(β[intact_vector, -2] - new_β[intact_vector, 0]) > 0.0:
                    steps_since_decohering = 0
                else:
                    steps_since_decohering = steps_since_decohering + 1

            # Now, combine both crack arrest conditions
            if steps_since_cracking > 20 and steps_since_decohering > 20:
                crack_arrested = True

            # Increment the time
            t = t + h
            t_vec.append(t)
            k = k + 1

            # Adapt the time-steps in case of success
            if successful_steps < 10:
                step_different = False
                successful_steps = successful_steps + 1
            else:
                if is_cracked:
                    # Already at the biggest step
                    if h == h_crack:
                        # Check if it's the step that we change h
                        if crack_step:
                            step_different = True
                            successful_steps = 0
                            crack_step = False
                        else:
                            step_different = False
                            successful_steps = successful_steps + 1
                    # Else make the step 1.5x bigger
                    else:
                        step_different = True
                        successful_steps = 0
                        if 1.5*h > h_max:
                            h = h_max
                        else:
                            h = 1.5*h
                else:
                    # Already at the biggest step
                    if h == h_max:
                        step_different = False
                        successful_steps = successful_steps + 1
                    # Else make the step 1.5x bigger
                    else:
                        step_different = True
                        successful_steps = 0
                        if 1.5*h > h_max:
                            h = h_max
                        else:
                            h = 1.5*h

    return F_vec, u, v, β, A, v_n, μ, λ, p_n, u_n, t_vec, Ψ_ε, Ψ_s, Ψ_K, work, contact_work


results = test(F, u, v, β, A, v_n, μ, λ, p_n, u_n, K, M, k_mask_vec, punctual_force_dof, h_max, t, Ψ_ε, Ψ_s, Ψ_K, work, contact_work)
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
t_vec = results[10]
Ψ_ε = results[11]
Ψ_s = results[12]
Ψ_K = results[13]
work = results[14]
contact_work = results[15]

# Now pickle the results (we wont take the displacements because the vtk has those)
pickle_folder = save_folder + "/Rhombus_hole/Rhombus_hole_" + mesh_angle + "_degrees/"
# Check that the folder exists and make it if not
if not os.path.isdir(pickle_folder):
    os.mkdir(pickle_folder)
np.save(pickle_folder + "F", F)
np.save(pickle_folder + "u", u)
np.save(pickle_folder + "v", v)
np.save(pickle_folder + "beta", β)
np.save(pickle_folder + "A", A)
np.save(pickle_folder + "v_n", v_n)
np.save(pickle_folder + "mu", μ)
np.save(pickle_folder + "lambda", λ)
np.save(pickle_folder + "p_n", p_n)
np.save(pickle_folder + "u_n", u_n)
np.save(pickle_folder + "t", t_vec)
np.save(pickle_folder + "strain_energy", Ψ_ε)
np.save(pickle_folder + "surface_energy", Ψ_s)
np.save(pickle_folder + "kinetic_energy", Ψ_K)
np.save(pickle_folder + "work", work)
np.save(pickle_folder + "contact_work", contact_work)
np.save(pickle_folder + "y_pos", y_pos)
np.save(pickle_folder + "dirichlet_dof", dirichlet_dof)
np.save(pickle_folder + "contact_dof", contact_dof)
np.save(pickle_folder + "punctual_force_dof", punctual_force_dof)
np.save(pickle_folder + "S", S)
