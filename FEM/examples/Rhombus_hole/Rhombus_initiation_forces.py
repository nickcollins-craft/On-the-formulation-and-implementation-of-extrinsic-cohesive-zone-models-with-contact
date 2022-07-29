import numpy as np
import sys
sys.path.append('../../src/')
from fem import *
from mesh import *

# Declare the name of the folder where the data is stored
data_folder = "/path/to/your/storage/folder/goes/here/"
# Declare the name of the general folder where the data will be saved (the data
# will be specifically saved in the sub folder corresponding to the angle)
save_folder = "/path/to/your/folder/goes/here/"

# Loop through the different meshes and recalculate the initiation force for each
for rhombus_angle in [70, 80, 90, 100, 110]:
    specific_data_folder = data_folder + str(rhombus_angle) + "_degrees/"
    t = np.load(specific_data_folder + "t.npy")
    β = np.load(specific_data_folder + "beta.npy")
    F = np.load(specific_data_folder + "F.npy")

    # Load in the mesh
    filename = "Rhombus_" + str(rhombus_angle) + "_degree"
    Rhombus_mesh = gmsh_mesh(filename)
    physical_name = Rhombus_mesh._physical_name

    # Re-find the punctual force nodes
    punctual_force_dof = []
    for fd in physical_name:
        if 'Applied displacement' in fd:
            for n in physical_name[fd]['nodes']:
                punctual_force_dof.extend([2*n + 1])

    # Find the point at which the first node breaks
    F_punctual_initiation = np.zeros((len(punctual_force_dof)))
    for time_step in range(len(t) - 1):
        if np.min(β[:, time_step]) <= 1e-12:
            t_stop = time_step
            break

    # Retain the force at this point
    F_punctual_initiation = F[punctual_force_dof, t_stop]

    # Now save the results
    np.save(specific_data_folder + "initiation_force", F_punctual_initiation)
