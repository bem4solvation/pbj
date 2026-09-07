import bempp_cl.api
from dolfinx.geometry import compute_colliding_cells, compute_collisions_points
import dolfinx
import numpy as np
from bempp_cl.api.assembly.blocked_operator import BlockedDiscreteOperator
from bempp_cl.api.assembly.discrete_boundary_operator import (
    InverseSparseDiscreteBoundaryOperator,
)
from bempp_cl.api.external import fenicsx
from scipy.sparse.linalg import LinearOperator
from numba import prange
import ufl
import time

"""Direct single-surface nonlinear Poisson-Boltzmann formulation.

This module assembles the standard two-field boundary-integral system for a single
solvent-solute interface using Laplace and modified Helmholtz operators.
"""


def verify_parameters(self):
    return True


def lhs(self):

    EI = self.ep_in + (self.ep_ex - self.ep_in) * self.Alpha
    KI = self.ep_ex * (self.kappa**2) * self.Alpha
    u = ufl.TrialFunction(self.fenics_space)
    v = ufl.TestFunction(self.fenics_space)

    I1 = bempp_cl.api.operators.boundary.sparse.identity(
        self.trace_space, self.bempp_space, self.bempp_space
    )  # 1
    mass = bempp_cl.api.operators.boundary.sparse.identity(
        self.bempp_space, self.bempp_space, self.trace_space
    )  # 1
    K1 = bempp_cl.api.operators.boundary.modified_helmholtz.double_layer(
        self.trace_space,
        self.bempp_space,
        self.bempp_space,
        self.kappa,
        assembler=self.operator_assembler,
    )  # K
    V1 = bempp_cl.api.operators.boundary.modified_helmholtz.single_layer(
        self.bempp_space,
        self.bempp_space,
        self.bempp_space,
        self.kappa,
        assembler=self.operator_assembler,
    )  # V

    A = [[None, None], [None, None]]
    trace_op = LinearOperator(self.trace_matrix.shape, lambda x: self.trace_matrix @ x)
    A_fem = fenicsx.FenicsOperator(
        (EI * ufl.inner(ufl.nabla_grad(u), ufl.nabla_grad(v)) + KI * u * v) * ufl.dx
    )
    A[0][0] = A_fem.weak_form()
    A[0][1] = -self.trace_matrix.T * self.ep_ex * mass.weak_form().to_sparse()
    A[1][0] = (0.5 * I1 - K1).weak_form() * trace_op
    A[1][1] = V1.weak_form()
    A = BlockedDiscreteOperator(np.array(A))
    self.matrices["A"] = A

    return


def rhs(self):

    rhs_fem_l = assemble_point_sources(self.fenics_space, self.x_q, 1 * self.q)
    rhs_bem_l = np.zeros(self.bempp_space.global_dof_count)
    self.rhs["rhs_1"], self.rhs["rhs_2"] = rhs_fem_l, rhs_bem_l


def block_diagonal_preconditioner(solute):

    from scipy.sparse import diags, csr_matrix
    from scipy.sparse.linalg import inv as sparse_inv

    P1 = diags(1.0 / solute.matrices["A"][0, 0].to_sparse().diagonal()).tocsr()

    identity = (
        bempp_cl.api.operators.boundary.sparse.identity(
            solute.bempp_space,
            solute.bempp_space,
            solute.bempp_space,
        )
        .weak_form()
        .to_sparse()
    )

    # scipy.sparse.bmat necesita matrices, no LinearOperator
    P2 = csr_matrix(sparse_inv(identity.tocsc()))

    Z12 = csr_matrix((P1.shape[0], P2.shape[1]))
    Z21 = csr_matrix((P2.shape[0], P1.shape[1]))

    solute.matrices["preconditioning_matrix_gmres"] = [
        [P1, Z12],
        [Z21, P2],
    ]

    solute.matrices["A_discrete"] = solute.matrices["A"]
    solute.rhs["rhs_discrete"] = np.concatenate(
        [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]
    )
    return


def block_diagonal_preconditioner_rhs(solute):
    return


def calculate_potential(simulation, rerun_all=False, rerun_rhs=False):

    start_time = time.time()
    if rerun_rhs and "A_discrete" in simulation.solutes[0].matrices:
        simulation.create_and_assemble_rhs()
    else:
        simulation.create_and_assemble_linear_system()

    simulation.timings["time_assembly"] = time.time() - start_time

    it_count = 0

    def count_iterations(x):
        nonlocal it_count
        it_count += 1
        if (it_count / 100) == (it_count // 100):
            print(it_count, x)

    # Solution by GMRES. FEM
    from scipy.sparse.linalg import gmres

    start1 = time.time()
    soln_l, info = gmres(
        simulation.matrices["A_discrete"],
        simulation.rhs["rhs_discrete"],
        M=simulation.matrices["preconditioning_matrix_gmres"],
        callback=count_iterations,
        rtol=simulation.gmres_tolerance_0,
        restart=simulation.gmres_restart,
    )  # Modificado
    end1 = time.time()

    # Time to solve the equation.
    curr_time1_L = end1 - start1
    Iter_l = it_count
    print(f"norm solution {np.linalg.norm(soln_l)}")
    print("Number of GMRES Lineal iterations: {0}".format(Iter_l))
    print("Total time in GMRES Lineal: {:5.2f} [s]".format(curr_time1_L))

    for index, solute in enumerate(simulation.solutes):
        if index > 0:
            raise NotImplementedError("Multiple solutes not implemented yet.")
        fem_size = solute.fenics_space.dofmap.index_map.size_global
        soln_fem_l = soln_l[:fem_size]
        u_l = dolfinx.fem.Function(solute.fenics_space)
        u_l.x.array[:] = np.ascontiguousarray(soln_fem_l)

        ep_in = solute.ep_in
        charges = np.ascontiguousarray(solute.q)
        charge_positions = np.ascontiguousarray(solute.x_q)

        @bempp_cl.api.complex_callable(jit=False)
        def U_c(x, n, domain_index, result):
            result[:] = (1 / (4.0 * np.pi * ep_in)) * np.sum(
                charges / np.linalg.norm(x - charge_positions, axis=1)
            )

        U_c0 = bempp_cl.api.GridFunction(solute.bempp_space0, fun=U_c)
        Um_l0 = Function_Um(solute.mesh0, u_l, solute.mesh_v)
        Um_l = bempp_cl.api.GridFunction(solute.bempp_space0, coefficients=Um_l0)
        rhs_0_values = rhs_0(solute, Um_l, U_c0)

        V0 = bempp_cl.api.operators.boundary.laplace.single_layer(
            solute.bempp_space0,
            solute.bempp_space0,
            solute.bempp_space0,
            assembler=solute.operator_assembler,
        )
        blocked_0 = V0.weak_form()  # 1x1 matrix.
        identity = bempp_cl.api.operators.boundary.sparse.identity(
            solute.bempp_space0, solute.bempp_space0, solute.bempp_space0
        ).weak_form()
        P_0 = InverseSparseDiscreteBoundaryOperator(
            identity
        )  # Mass Matrix 1x1 preconditioner.

        # Solution by GMRES.
        it_count = 0
        start1 = time.time()
        Sol_l, info = gmres(
            blocked_0,
            rhs_0_values,
            M=P_0,
            callback=count_iterations,
            rtol=solute.gmres_tolerance_0,
            restart=solute.gmres_restart,
        )
        end1 = time.time()
        dUm_l = bempp_cl.api.GridFunction(solute.bempp_space0, coefficients=Sol_l)
        # Time to solve the equation.
        curr_time1 = end1 - start1
        print(f"norm solution {np.linalg.norm(Sol_l)}")
        print("Total time in GMRES BEM: {:5.2f} [s]".format(curr_time1))
        print("Number of GMRES iterations of dU_m: {0}".format(it_count))

        solute.results["phi"] = Um_l
        solute.results["d_phi"] = dUm_l

    return


def rhs_0(solute, Um_l, U_c0):

    I0 = bempp_cl.api.operators.boundary.sparse.identity(
        solute.bempp_space0, solute.bempp_space0, solute.bempp_space0
    )  # 1
    K0 = bempp_cl.api.operators.boundary.laplace.double_layer(
        solute.bempp_space0,
        solute.bempp_space0,
        solute.bempp_space0,
        assembler=solute.operator_assembler,
    )  # K
    rhs_0 = ((0.5 * I0 + K0) * Um_l - U_c0).projections(solute.bempp_space0)

    return rhs_0


def assemble_point_sources(space, points, weights):
    vector = np.zeros(space.dofmap.index_map.size_global)

    mesh = space.mesh
    tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
    for point, weight in zip(points, weights):
        # Get cell
        cell = locate_cell(point, space.mesh, tree)

        # Get local coordinates
        # Note: this currently only works for affine triangles and tetrahedra
        v = [mesh.geometry.x[i] for i in mesh.geometry.dofmaps[0][cell]]
        local_coordinates = get_local_coordinates(v, point)

        # Note: this currently only works for scalar-valued elements that use an identity push forward map
        values = space.element.basix_element.tabulate(
            0, np.array([local_coordinates], dtype=np.float64)
        )[0, 0, :, 0]
        dofs = space.dofmap.cell_dofs(cell)

        for d, v in zip(dofs, values):
            vector[d] += v * weight

    return vector


def locate_cell(point, mesh, tree):
    """
    Encuentra que tetrahedro contiene al punto ingresado

    point = Arreglo 3x1 con el punto buscado.
    mesh = Malla volumétrica donde buscar.

    Retorna:
        cell = N° del tetrahedro que contiene el punto
    """
    cell_candidates = compute_collisions_points(tree, point)
    cell = compute_colliding_cells(mesh, cell_candidates, point).array[0]

    return cell


def get_local_coordinates(vertices, point):
    """Get the local coordinates of the point in the cell."""
    origin = vertices[0]
    axes = [v - origin for v in vertices[1:]]
    tdim = 3
    if len(axes) == 2:
        axes.append(np.cross(axes[0], axes[1]))
        tdim = 2

    assert len(axes) == 3

    return np.linalg.solve(np.array(axes).T, point - origin)[:tdim]


def numba_classify(signed_distances):
    """This function simply iterates through a list of distances in parallel and labels them. Saves a little bit of memory."""
    label = np.zeros_like(signed_distances, dtype=np.uint8)
    for i in prange(label.shape[0]):
        if signed_distances[i] > 0:
            label[i] = 1
        elif signed_distances[i] < 0:
            label[i] = 0
        else:
            label[i] = 2
    return label


# Function to calculate the potential on the surface.
def Function_Um(grid, u, mesh):
    points_verts = np.array(grid.vertices.T, dtype=np.float64)
    Um = np.zeros(len(points_verts))
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    cells, points_on_proc = [], []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points_verts)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, cell_candidates, points_verts
    )
    for i, point in enumerate(points_verts):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    Um0 = u.eval(points_on_proc, cells)
    for i in range(len(Um0)):
        Um[i] = Um0[i][0]
    return np.array(Um)
