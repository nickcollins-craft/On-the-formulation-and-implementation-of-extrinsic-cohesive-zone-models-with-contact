import numpy as np


# Define the bulk material
class material:
    # Initialise its value with the Young's modulus, Poisson's ratio, a density,
    # and whether we make a plane stress or plane strain assumption
    def __init__(self, E, ν, ρ, load_type='plane_stress'):
        self._E = E
        self._ν = ν
        self._ρ = ρ
        self._load_type = load_type

    # Define the element stiffness tensor for 2D, depending on whether it's plane
    # strain or plane stress. This matrix acts on a vector [ε_xx, ε_yy, γ_xy]
    # to return a vector [σ_xx, σ_yy, σ_xy]
    def D(self, dim):
        if (dim == 2):
            D = np.zeros((3, 3))
            if self._load_type == 'plane_strain':
                coef = self._E/((1 + self._ν)*(1 - 2*self._ν))
                # Plain strain matrix
                D[0, 0] = coef*(1.0 - self._ν)
                D[0, 1] = coef*self._ν
                D[0, 2] = 0.0

                D[1, 0] = D[0, 1]
                D[1, 1] = D[0, 0]
                D[1, 2] = 0.0

                D[2, 0] = 0.0
                D[2, 1] = 0.0
                D[2, 2] = coef*(1.0 - 2*self._ν)/2
            elif self._load_type == 'plane_stress':
                coef = self._E/(1 - self._ν**2)
                # Plain stress matrix
                D[0, 0] = coef
                D[0, 1] = coef*self._ν
                D[0, 2] = 0.0

                D[1, 0] = D[0, 1]
                D[1, 1] = D[0, 0]
                D[1, 2] = 0.0

                D[2, 0] = 0.0
                D[2, 1] = 0.0
                D[2, 2] = coef*(1.0 - self._ν)/2
            else:
                # If it's not one of the two allowable load types, raise an error
                raise ValueError('self._load_type', 'load type not implemented, please enter plane_strain or plane_stress')
        else:
            # If in 3D, raise an error
            raise ValueError('dim', 'dimension not implemented')
        return D


# Define the finite element as either T3 or Q4
class fem_element:
    # Allocate the type of finite element, and the number of nodes and dimensions
    def __init__(self, fem_type):
         self._fem_type = fem_type
         if self._fem_type == 'T3':
             self._n_nodes = 3
             self._dim = 2
         elif self._fem_type == 'Q4':
             self._n_nodes = 4
             self._dim = 2
         else:
             # If it's not T3 or Q4, raise an error
             raise ValueError('self._fem_type', 'Element type not implemented')

    # Define the appropriate shape functions for T3 or Q4 in the internal coordinates
    # ξ and η, given those internal coordinates
    def shape_functions(self, ξ, η):
        if self._fem_type == 'T3':
            N = np.zeros(3)
            N[0] = 1.0 - ξ - η
            N[1] = ξ
            N[2] = η
        elif self._fem_type == 'Q4':
            N = np.zeros(4)
            N[0] = (1./4.)*(1 + ξ)*(1 + η)
            N[1] = (1./4.)*(1 - ξ)*(1 + η)
            N[2] = (1./4.)*(1 - ξ)*(1 - η)
            N[3] = (1./4.)*(1 + ξ)*(1 - η)
        return N

    # Define the derivatives of the shape functions w.r.t. the internal coordinates
    def shape_functions_derivatives(self, ξ, η):
        if self._fem_type == 'T3':
            Nξ = np.zeros(3)
            Nξ[0] = -1.0
            Nξ[1] = 1.0
            Nξ[2] = 0.0
            Nη = np.zeros(3)
            Nη[0] = -1.0
            Nη[1] = 0.0
            Nη[2] = 1.0
        elif self._fem_type == 'Q4':
            Nξ = np.zeros(4)
            Nξ[0] = (1./4.)*(1 + η)
            Nξ[1] = (-1./4.)*(1 + η)
            Nξ[2] = (-1./4.)*(1 - η)
            Nξ[3] = (1./4.)*(1 - η)
            Nη = np.zeros(4)
            Nη[0] = (1./4.)*(1 + ξ)
            Nη[1] = (1./4.)*(1 - ξ)
            Nη[2] = (-1./4.)*(1 - ξ)
            Nη[3] = (-1./4.)*(1 + ξ)
        return Nξ, Nη


# Define the finite element model as a whole
class fem_model:
    # Load up the mesh
    def __init__(self, mesh, thickness=None, mass_type='consistent', integration_type='four_point'):
        self._mesh = mesh
        self._thickness = thickness
        self._mass_type = mass_type
        self._integration_type = integration_type
        # Compute the number of degrees of freedom (dof)
        n_dof = 0
        for n in mesh.points:
            # Every node has 2 dof in 2D
            n_dof = n_dof + 2
        self._n_dof = n_dof
        # Initialise the stiffness matrix
        self.K = np.zeros((self._n_dof, self._n_dof))
        # Initialise the mass matrix
        self.M = np.zeros((self._n_dof, self._n_dof))

        # Work out if we have triangles or quads (this code assumes exclusively
        # one or the other), and allocate the correct finite element type
        for mesh_indexer in range(len(self._mesh.cells)):
            if self._mesh.cells[mesh_indexer].type == 'triangle':
                self._fem_type = 'T3'
            elif self._mesh.cells[mesh_indexer].type == 'quad':
                self._fem_type = 'Q4'
            elif self._mesh.cells[mesh_indexer].type != 'vertex' and self._mesh.cells[mesh_indexer].type != 'line':
                print("mesh element type = ", self._mesh.cells[mesh_indexer].type)
                raise ValueError('self._mesh.cells[mesh_indexer].type', 'Element type not implemented')

    # Report the number of dof and the stiffness and mass matrices
    def __str__(self):
        message = """
        number of dof = {0}
        stiffness matrix = {1}
        mass matrix = {2}
        """.format(self._n_dof, self.K, self.M)
        return message

    # Return the location of the Gauss points in internal coordinates, plus weighting factors
    def gauss_points(self, e):
        if self._fem_type == 'T3':
            ξ_coordinate = [1./3.]
            η_coordinate = [1./3.]
            weighting_factor = [0.5]
        elif self._fem_type == 'Q4':
            if self._integration_type == 'four_point':
                ξ_coordinate = [-1./np.sqrt(3), 1./np.sqrt(3), -1./np.sqrt(3), 1./np.sqrt(3)]
                η_coordinate = [-1./np.sqrt(3),-1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3)]
                weighting_factor = [1., 1., 1., 1.]
            elif self._integration_type == 'one_point':
                ξ_coordinate = [0.]
                η_coordinate = [0.]
                weighting_factor = [4.]
            else:
                raise ValueError('self._integration_type', 'Gaussian quadrature rule not implemented, for Q4 element it should be one_point or four_point')
        return ξ_coordinate, η_coordinate, weighting_factor

    # Define the formula to calculate the area of an element, given the coordinates
    # of its nodes
    def element_area(self, e):
        if self._fem_type == 'T3':
            fem_e = fem_element(self._fem_type)
            x = []
            y = []
            # Loop through each node in the element
            for i in range(fem_e._n_nodes):
                # Get the global node number
                n = e[i]
                # Get the x and y coordinates of that node
                x = np.append(x, self._mesh.points[n][0])
                y = np.append(y, self._mesh.points[n][1])
            # Once all coordinates are obtained, calculate the area of the element
            A = 0.5*abs(x[0]*(y[1] - y[2]) + x[1]*(y[2] - y[0]) + x[2]*(y[0] - y[1]))
        elif self._fem_type == 'Q4':
            fem_e = fem_element(self._fem_type)
            x = []
            y = []
            # Loop through each node in the element
            for i in range(fem_e._n_nodes):
                # Get the global node number
                n = e[i]
                # Get the x and y coordinates of that node
                x = np.append(x, self._mesh.points[n][0])
                y = np.append(y, self._mesh.points[n][1])
            # Get the side and diagonal lengths (note that we don't a priori know
            # if the nodes have been given to us in a cyclic order, so we need
            # to check and arrange conveniently)
            lengths = np.zeros((6))
            lengths[0] = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
            lengths[1] = np.sqrt((x[2] - x[1])**2 + (y[2] - y[1])**2)
            lengths[2] = np.sqrt((x[3] - x[2])**2 + (y[3] - y[2])**2)
            lengths[3] = np.sqrt((x[0] - x[3])**2 + (y[0] - y[3])**2)
            lengths[4] = np.sqrt((x[2] - x[0])**2 + (y[2] - y[0])**2)
            lengths[5] = np.sqrt((x[3] - x[1])**2 + (y[3] - y[1])**2)
            # Find the longest entry (major diagonal), which we will label f, and
            # then label the other diagonal e. The first of the two points not connected
            # by f will be point A, with sides a and d. Sides b and c follow naturally
            if np.argmax(lengths) == 5:
                f_diag = lengths[5]
                e_diag = lengths[4]
                a_side = lengths[0]
                d_side = lengths[3]
                b_side = lengths[1]
                c_side = lengths[2]
            elif np.argmax(lengths) == 4:
                f_diag = lengths[4]
                e_diag = lengths[5]
                a_side = lengths[1]
                d_side = lengths[0]
                b_side = lengths[2]
                c_side = lengths[3]
            elif np.argmax(lengths) == 3:
                f_diag = lengths[3]
                e_diag = lengths[1]
                a_side = lengths[5]
                d_side = lengths[0]
                b_side = lengths[2]
                c_side = lengths[4]
            elif np.argmax(lengths) == 2:
                f_diag = lengths[2]
                e_diag = lengths[0]
                a_side = lengths[4]
                d_side = lengths[3]
                b_side = lengths[1]
                c_side = lengths[5]
            elif np.argmax(lengths) == 1:
                f_diag = lengths[1]
                e_diag = lengths[3]
                a_side = lengths[0]
                d_side = lengths[4]
                b_side = lengths[5]
                c_side = lengths[2]
            else:
                f_diag = lengths[0]
                e_diag = lengths[2]
                a_side = lengths[4]
                d_side = lengths[1]
                b_side = lengths[3]
                c_side = lengths[5]
            # Well-arranged, proceed with Bretschneider's formula
            A = (1/4)*np.sqrt((2*e_diag*f_diag)**2 - (b_side**2 + d_side**2 - a_side**2 - c_side**2)**2)
            if type(A) != np.float64:
                raise ValueError('type(A)', 'Area of quad element is not a positive real float, something is wrong in the formulation')
        return A

    # Compute the Jacobian "J" matrix of an individual finite element e at a given
    # Gauss point p
    def compute_elementary_Jacobian_matrix(self, e, p):
        fem_e = fem_element(self._fem_type)
        n_dof_e = fem_e._n_nodes*fem_e._dim
        p_ξ = p[0]
        p_η = p[1]
        N = fem_e.shape_functions(p_ξ, p_η)
        Nξ, Nη = fem_e.shape_functions_derivatives(p_ξ, p_η)

        # Compute Jacobian matrix
        J = np.zeros((fem_e._dim, fem_e._dim))
        # Loop through each node and calculate its contribution to the Jacobian
        for i in range(fem_e._n_nodes):
            n = e[i]
            x = self._mesh.points[n][0]
            y = self._mesh.points[n][1]
            J[0, 0] = J[0, 0] + Nξ[i]*x
            J[0, 1] = J[0, 1] + Nξ[i]*y
            J[1, 0] = J[1, 0] + Nη[i]*x
            J[1, 1] = J[1, 1] + Nη[i]*y
        return J

    # Compute the "B" matrix of an individual finite element (which will give us
    # the finite element estimate of the strain), where e is the element number, and p the Gauss points
    def compute_elementary_B_matrix(self, e, p):
        fem_e = fem_element(self._fem_type)
        n_dof_e = fem_e._n_nodes*fem_e._dim
        p_ξ = p[0]
        p_η = p[1]
        N = fem_e.shape_functions(p_ξ, p_η)
        Nξ, Nη = fem_e.shape_functions_derivatives(p_ξ, p_η)

        # Compute inverse of Jacobian matrix
        J_inv = np.linalg.inv(self.compute_elementary_Jacobian_matrix(e, p))

        # Compute the derivative w.r.t x and y of the shape function at each node
        Nx = np.zeros(fem_e._n_nodes)
        Ny = np.zeros(fem_e._n_nodes)
        # For each node in the element, calculate the contribution to the derivative
        for i in range(fem_e._n_nodes):
             Nx[i] = J_inv[0, 0]*Nξ[i] + J_inv[0, 1]*Nη[i]
             Ny[i] = J_inv[1, 0]*Nξ[i] + J_inv[1, 1]*Nη[i]

        # Construct the B matrix (its form is consistent with the choice of the
        # representation of strain)
        B = np.zeros((3, n_dof_e))
        # Loop through each node in the element and use the previously calculated
        # derivatives to construct "B"
        for i in range(fem_e._n_nodes):
            B[0, 2*i] = Nx[i]
            B[1, 2*i] = 0.0
            B[2, 2*i] = Ny[i]
            B[0, 2*i + 1] = 0.0
            B[1, 2*i + 1] = Ny[i]
            B[2, 2*i + 1] = Nx[i]
        return B

    # Define the stiffness matrix for an element, given the element and the material
    # stiffness tensor
    def compute_elementary_stiffness_matrix(self, e, D):
        # Get the element type and proceed accordingly
        fem_e = fem_element(self._fem_type)
        n_dof_e = fem_e._n_nodes*fem_e._dim
        # Loop over the Gauss points and fill in the stiffness matrix.
        K_e = np.zeros((n_dof_e, n_dof_e))
        ξ_coordinate, η_coordinate, weighting_factor = self.gauss_points(e)
        for gauss_point_index in range(len(ξ_coordinate)):
            p = [ξ_coordinate[gauss_point_index], η_coordinate[gauss_point_index]]
            # Get the elementary B matrix and Jacobian determinant from the
            # previous procedures
            det_J = np.linalg.det(self.compute_elementary_Jacobian_matrix(e, p))
            B = self.compute_elementary_B_matrix(e, p)
            # Fill in the stiffness matrix via multiplication of the appropriate
            # terms
            K_e = K_e + weighting_factor[gauss_point_index]*det_J*self._thickness*np.dot(B.T, np.dot(D, B))
        return K_e

    # Define the consistent mass matrix for an element, given the element and the density
    def compute_elementary_mass_matrix(self, e, ρ):
        fem_e = fem_element(self._fem_type)
        n_dof_e = fem_e._n_nodes*fem_e._dim
        if self._mass_type == 'lumped':
            A = self.element_area(e)
            M_e = (ρ*A*self._thickness/fem_e._n_nodes)*np.eye(n_dof_e)
        elif self._mass_type == 'consistent':
            M_e = np.zeros((n_dof_e, n_dof_e))
            ξ_coordinate, η_coordinate, weighting_factor = self.gauss_points(e)
            for gauss_point_index in range(len(ξ_coordinate)):
                N = fem_e.shape_functions(ξ_coordinate[gauss_point_index], η_coordinate[gauss_point_index])
                p = [ξ_coordinate[gauss_point_index], η_coordinate[gauss_point_index]]
                det_J = np.linalg.det(self.compute_elementary_Jacobian_matrix(e, p))
                H = np.zeros((fem_e._dim, n_dof_e))
                for i in range(len(N)):
                    H[0, 2*i] = N[i]
                    H[1, 2*i + 1] = N[i]
                M_e = M_e + ρ*self._thickness*weighting_factor[gauss_point_index]*np.matmul(np.transpose(H), H)*det_J
        else:
            # If it's not one of the two allowable mass matrix types, raise an error
            raise ValueError('self._mass_type', 'Mass matrix type not implemented, please enter consistent or lumped')
        return M_e

    # Define the stiffness matrix at a structural level, given an element and its
    # stiffness matrix
    def assemble_stiffness_matrix(self, e, K_e):
        # Initialse the dof list and the node counter
        dof_index_n = []
        n_cnt = 0
        # Loop through the nodes in the element
        for n in e:
            # Specify the global dofs of the node we consider
            dof_index_n = [2*n, 2*n + 1]
            # Initialise the node counter for the other nodes that are going to
            # contribute stiffness to that dof
            m_cnt = 0
            # Loop through the nodes (again)
            for m in e:
                # Specify the global dofs of the node we consider
                dof_index_m = [2*m, 2*m + 1]
                # Loop through the local dofs
                for i in range(2):
                    for j in range(2):
                        # The stiffness terms contributed to the dofs at node n
                        # by the dofs at node m. Each pair gives a 2x2 contribution
                        # that is mapped into the global coordinates, and added to
                        # whatever is already contained in the global stiffness matrix.
                        self.K[dof_index_n[i], dof_index_m[j]] = self.K[dof_index_n[i], dof_index_m[j]] + K_e[i + n_cnt*2, j + m_cnt*2]
                m_cnt = m_cnt + 1
            n_cnt = n_cnt + 1
        return

    # Define the mass matrix at a structural level, given an element and its mass
    # matrix
    def assemble_mass_matrix(self, e, M_e):
        # Initialse the dof list and the node counter
        dof_index_n = []
        n_cnt = 0
        # Loop through the nodes in the element
        for n in e:
            # Specify the global dofs of the node we consider
            dof_index_n = [2*n, 2*n + 1]
            # Initialise the node counter for the other nodes that are
            # going to contribute mass to that dof
            m_cnt = 0
            # Loop through the nodes (again)
            for m in e:
                # Specify the global dofs of the node we consider
                dof_index_m = [2*m, 2*m + 1]
                # Loop through the local dofs
                for i in range(2):
                    for j in range(2):
                        # The mass terms contributed to the dofs at node n
                        # by the dofs at node m. Each pair gives a 2x2 contribution
                        # that is mapped into the global coordinates, and added to
                        # whatever is already contained in the global mass matrix.
                        self.M[dof_index_n[i], dof_index_m[j]] = self.M[dof_index_n[i], dof_index_m[j]] + M_e[i + n_cnt*2, j + m_cnt*2]
                m_cnt = m_cnt + 1
            n_cnt = n_cnt + 1
        return

    # Get the full global stiffness matrix by looping over all elements
    def compute_stiffness_matrix(self, material):
        if self._fem_type == 'T3':
            # Loop over the triangle elements
            for e in self._mesh.cells_dict['triangle']:
                D = material.D(2)
                K_e = self.compute_elementary_stiffness_matrix(e, D)
                K = self.assemble_stiffness_matrix(e, K_e)
        elif self._fem_type == 'Q4':
            # Loop over the quad elements
            for e in self._mesh.cells_dict['quad']:
                D = material.D(2)
                K_e = self.compute_elementary_stiffness_matrix(e, D)
                K = self.assemble_stiffness_matrix(e, K_e)
        else:
            raise ValueError('self._fem_type', 'Element type not implemented')
        return K

    # Get the full global mass matrix by looping over all elements
    def compute_mass_matrix(self, material):
        if self._fem_type == 'T3':
            # Loop over the triangle elements
            for e in self._mesh.cells_dict['triangle']:
                M_e = self.compute_elementary_mass_matrix(e, material._ρ)
                self.assemble_mass_matrix(e, M_e)
        elif self._fem_type == 'Q4':
            # Loop over the quad elements
            for e in self._mesh.cells_dict['quad']:
                M_e = self.compute_elementary_mass_matrix(e, material._ρ)
                self.assemble_mass_matrix(e, M_e)
        else:
            raise ValueError('self._fem_type', 'Element type not implemented')
        return

    # Give the method for computing the strain at the Gauss points (not the nodes)
    def compute_strain_at_gauss_points(self, u):
        if self._fem_type == 'T3':
            fem_e = fem_element(self._fem_type)
            # Loop over the triangle elements
            ε = []
            for e in self._mesh.cells_dict['triangle']:
                # Get the displacement of the element
                u_e = []
                for n in e:
                    u_e.extend([u[2*n], u[2*n + 1]])
                # Get the Gauss point information and calculate the strain
                ξ_coordinate, η_coordinate, weighting_factor = self.gauss_points(e)
                p = [ξ_coordinate[0], η_coordinate[0]]
                B = self.compute_elementary_B_matrix(e, p)
                ε.extend(np.dot(B, u_e))
        elif self._fem_type == 'Q4':
            fem_e = fem_element(self._fem_type)
            # Loop over the quad elements
            ε = []
            for e in self._mesh.cells_dict['quad']:
                # Get the displacement of the element
                u_e = []
                for n in e:
                    u_e.extend([u[2*n], u[2*n + 1]])
                # Loop over the Gauss points, and calculate the strain at each one
                ξ_coordinate, η_coordinate, weighting_factor = self.gauss_points(e)
                ε_GP = np.zeros((len(ξ_coordinate), 3))
                for gauss_point_index in range(len(ξ_coordinate)):
                    p = [ξ_coordinate[gauss_point_index], η_coordinate[gauss_point_index]]
                    B = self.compute_elementary_B_matrix(e, p)
                    # Calculate the strain at that Gauss point
                    ε_GP[gauss_point_index, :] = np.dot(B, u_e)
                # Calculate a strain value for the cell by averaging the value over
                # the Gauss points
                ε.extend(np.mean(ε_GP, axis=0))
        return ε
