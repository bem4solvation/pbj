import bempp_cl.api
from dataclasses import dataclass
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
from scipy.sparse.linalg import gmres
import ufl
import time
import math

"""Direct single-surface nonlinear Poisson-Boltzmann formulation.

This module assembles the standard two-field boundary-integral system for a single
solvent-solute interface using Laplace and modified Helmholtz operators.
"""


qe = 1.60217646e-19  # Charge of an electron [C]
Na = 6.0221415e23  # Avogadro constant [1/mol]
E0 = 8.854187818e-12  # Vacuum permittivity [C^2/Jm]
m2A = 10**10  # Transformation of length units from meter to Angstrom [A/m]
Kcal2J = 4184  # Transformation of energy units from kilocalories to Joule [Kcal/J]
KB = 1.380649e-23  # Boltzmann constant [J/K]
T = 298.15  # Absolute temperature [K]
C1 = (m2A * (qe**2)) / (KB * T * E0)


@dataclass
class NonlinearState:
    soln_l: object
    soln0_nl: object
    d_soln_nl: object
    d_soln0_nl: object


def verify_parameters(self):
    return True


def lhs(self):

    from scipy.sparse import csr_matrix

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
    A_blocked = BlockedDiscreteOperator(np.array(A))
    self.matrices["A"] = A_blocked

    A_nl = A
    B = fenicsx.FenicsOperator(
        (EI * ufl.inner(ufl.nabla_grad(u), ufl.nabla_grad(v))) * ufl.dx
    )
    A_nl[0][0] = B.weak_form()
    A_nl_blocked = BlockedDiscreteOperator(np.array(A_nl))
    self.matrices["A_nl"] = A_nl_blocked
    self.matrices["B"] = B.weak_form()

    V0 = bempp_cl.api.operators.boundary.laplace.single_layer(
        self.bempp_space0,
        self.bempp_space0,
        self.bempp_space0,
        assembler=self.operator_assembler,
    )

    self.matrices["A0"] = V0.weak_form()

    ZM1 = bempp_cl.api.ZeroBoundaryOperator(
        self.bempp_space, self.bempp_space, self.trace_space
    )  # 0
    ZK1 = bempp_cl.api.ZeroBoundaryOperator(
        self.trace_space, self.bempp_space, self.bempp_space
    )  # 0
    ZV1 = bempp_cl.api.ZeroBoundaryOperator(
        self.bempp_space, self.bempp_space, self.bempp_space
    )  # 0
    trace_op = LinearOperator(self.trace_matrix.shape, lambda x: self.trace_matrix @ x)

    blocks_N0 = [
        [None, None],
        [None, None],
    ]  # Later the term blocks_dN[0][0] is generated
    blocks_N0[0][1] = csr_matrix(self.trace_matrix.T * ZM1.weak_form().to_sparse())
    blocks_N0[1][0] = ZK1.weak_form() * trace_op
    blocks_N0[1][1] = ZV1.weak_form()
    self.matrices["dA_nl"] = blocks_N0

    return


def rhs(self):

    rhs_fem_l = assemble_point_sources(self.fenics_space, self.x_q, C1 * self.q)
    rhs_bem_l = np.zeros(self.bempp_space.global_dof_count)
    self.rhs["rhs_1"], self.rhs["rhs_2"] = rhs_fem_l, rhs_bem_l


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


def block_diagonal_preconditioner(solute):

    from scipy.sparse import diags, csr_matrix
    from scipy.sparse.linalg import inv as sparse_inv

    P1 = diags(1.0 / solute.matrices["B"].to_sparse().diagonal()).tocsr()

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


def calculate_potential(simulation, rerun_all=False, rerun_rhs=False):

    from bempp_cl.api.linalg.iterative_solvers import IterationCounter

    callback = IterationCounter(True)

    if rerun_all and rerun_rhs:  # if both are True, just rerun_all
        rerun_rhs = False

    elif ("phi" not in simulation.solutes[0].results) or (rerun_all) or (rerun_rhs):
        start_time = time.time()
        if rerun_rhs and "A_discrete" in simulation.solutes[0].matrices:
            simulation.create_and_assemble_rhs()
        else:
            simulation.create_and_assemble_linear_system()

        simulation.timings["time_assembly"] = time.time() - start_time

        if len(simulation.solutes) > 1:
            raise NotImplementedError("Multiple solutes not implemented yet.")

        start1 = time.time()
        # Solution FEM-BEM linear PB
        soln_l, info = gmres(
            simulation.matrices["A_discrete"],
            simulation.rhs["rhs_discrete"],
            M=simulation.matrices["preconditioning_matrix_gmres"],
            rtol=simulation.gmres_tolerance_0,
            restart=simulation.gmres_restart,
            callback=callback,
            callback_type="legacy",
        )
        simulation.timings["time_gmres_lpbe"] = time.time() - start1
        simulation.run_info["solver_iteration_count_lpbe"] = callback.count

        solute = simulation.solutes[0]
        fem_size = solute.fenics_space.dofmap.index_map.size_global
        bem_size = solute.bempp_space.global_dof_count

        u_l = dolfinx.fem.Function(solute.fenics_space)
        u_l.x.array[:] = np.ascontiguousarray(soln_l[:fem_size])
        solute.results["phi_fem_l"] = u_l / C1
        d_phi_s_l = bempp_cl.api.GridFunction(
            solute.bempp_space,
            coefficients=soln_l[fem_size:].real,
        )
        phi_s_l = bempp_cl.api.GridFunction(
            solute.bempp_space,
            coefficients=function_Um(solute.trace_space.grid, u_l, solute.mesh_v).real,
        )
        solute.results["phi_s_l"] = phi_s_l / C1
        solute.results["d_phi_s_l"] = d_phi_s_l / C1

        ep_in = solute.ep_in
        q = np.ascontiguousarray(solute.q)
        x_q = np.ascontiguousarray(solute.x_q)

        @bempp_cl.api.complex_callable(jit=False)
        def U_c0(x, n, domain_index, result):
            result[:] = (C1 / (4.0 * np.pi * ep_in)) * np.sum(
                q / np.linalg.norm(x - x_q, axis=1)
            )

        U_c = bempp_cl.api.GridFunction(solute.bempp_space0, fun=U_c0)
        Um_l = bempp_cl.api.GridFunction(
            solute.bempp_space0,
            coefficients=function_Um(solute.mesh, u_l, solute.mesh_v).real,
        )
        rhs_0_values = rhs_0(solute, Um_l, U_c)

        identity = bempp_cl.api.operators.boundary.sparse.identity(
            solute.bempp_space0, solute.bempp_space0, solute.bempp_space0
        ).weak_form()
        P_0 = InverseSparseDiscreteBoundaryOperator(identity)

        # Solution by GMRES for d_phi LPBE.
        start1 = time.time()
        sol, info = gmres(
            solute.matrices["A0"],
            rhs_0_values,
            M=P_0,
            rtol=solute.gmres_tolerance_0,
            restart=solute.gmres_restart,
            callback=callback,
            callback_type="legacy",
        )
        simulation.timings["time_gmres_lpbe_dphi"] = time.time() - start1
        simulation.run_info["solver_iteration_count_lpbe_dphi"] = callback.count
        dUm_l = bempp_cl.api.GridFunction(solute.bempp_space0, coefficients=sol.real)

        solute.results["phi_l"] = (
            Um_l / C1
        )  # using C1 to convert from nondimensional to PyGBe units
        solute.results["d_phi_l"] = (
            dUm_l / C1
        )  # using C1 to convert from nondimensional to PyGBe units
        solute.results["phi_coul"] = U_c / C1

        # Newton-Raphson iteration
        eps = 100.0  # initial error measure
        gmres_tol = solute.gmres_tolerance_0
        Iter = 0  # Iteration counter
        Taylor_expansion = solute.taylor_expansion
        u0_nl = dolfinx.fem.Function(solute.fenics_space)
        u0_nl.x.array[:] = 0.0
        soln0_nl = np.concatenate([np.zeros(fem_size), np.zeros(bem_size)])
        d_soln0_nl = soln0_nl
        d_zeros = soln0_nl
        c_bem = np.zeros(bem_size)
        KI = solute.ep_ex * (solute.kappa**2) * solute.Alpha
        v = ufl.TestFunction(solute.fenics_space)

        A_nl = solute.matrices["A_nl"]
        state = NonlinearState(
            soln_l=soln_l,
            soln0_nl=soln0_nl,
            d_soln_nl=d_soln0_nl,
            d_soln0_nl=d_soln0_nl,
        )

        simulation.timings["time_gmres_npbe_it"] = []
        simulation.timings["time_npbe_it"] = []
        simulation.run_info["solver_iteration_count_npbe_it"] = []
        simulation.run_info["solver_term_npbe"] = []
        simulation.run_info["solver_error_npbe"] = []

        # Start of the nonlinear algorithm
        while (eps > simulation.nonlinear_tol) and (Iter < solute.nonlinear_maxiter):
            start2 = time.time()
            Iter += 1
            simulation.run_info["solver_term_npbe"].append(Taylor_expansion)
            # Section 1: Choosing the Taylor approximation of vector c.
            NL_Fem_G_S, NL_Fem_G_C = Taylor_Expansion_of_vector_c(
                Taylor_expansion, u0_nl, u_l, solute.fenics_space
            )

            # Section 2: Update the new right-hand side vector.
            c_fem = dolfinx.fem.assemble_vector(
                dolfinx.fem.form(KI * NL_Fem_G_S * v * ufl.dx)
            ).array
            # The combination of rhs in Ωi of FEM.
            c_nlG = np.concatenate([c_fem, c_bem])
            rhs_nlG = -(
                A_nl * (soln_l + state.soln0_nl)
                - simulation.rhs["rhs_discrete"]
                + c_nlG
            )

            # Creation of the matrix N.
            ud = ufl.TrialFunction(solute.fenics_space)
            dA_nl = solute.matrices["dA_nl"]
            dA_nl[0][0] = (
                fenicsx.FenicsOperator(KI * NL_Fem_G_C * ud * v * ufl.dx)
            ).weak_form()
            dA_nl = BlockedDiscreteOperator(np.array(dA_nl))

            # Section 3: Solve the nonlinear matrix system with GMRES.
            # Solution by GMRES.
            callback = IterationCounter(True)
            start1 = time.time()
            d_soln_nl, info = gmres(
                (A_nl + dA_nl),
                rhs_nlG,
                x0=d_zeros,  # pr
                M=simulation.matrices["preconditioning_matrix_gmres"],
                rtol=gmres_tol,
                restart=solute.gmres_restart,
                callback=callback,
                callback_type="legacy",
            )
            state.d_soln_nl = d_soln_nl
            simulation.timings["time_gmres_npbe_it"].append(time.time() - start1)
            simulation.run_info["solver_iteration_count_npbe_it"].append(callback.count)

            # Choosing the scheme to solve the problem and the next Taylor expansion.
            if Iter == 1:
                (
                    Scheme,
                    Taylor_expansion_list,
                    w0_NR,
                    Iter_Transition,
                    Bisection_Secant_Method,
                ) = Scheme_election(state, simulation, solute, c_bem)
            if Iter <= Iter_Transition:
                Taylor_expansion = Taylor_expansion_list[Iter - 1]
            # Section 4: Calculate the relaxation factor of the next iteration by Newton-Raphson method total.
            if Iter == 1:
                if Bisection_Secant_Method:
                    _, I_w0_NR, w0_NR = w_optimal_by_Bisection(
                        state,
                        simulation,
                        solute,
                        c_bem,
                        Taylor_expansion,
                        2,
                        3,
                        0,
                        Secant_equation=True,
                        Tol_w=simulation.omega_tol,
                    )
                    # print(
                    #     "Iter Total BI-SEC previus w I_w0_NR=%d: w0_NR=%g"
                    #     % (I_w0_NR, w0_NR)
                    # )
                w, Iter_w = w_optimal_by_Newton_Rapson(
                    state,
                    simulation,
                    solute,
                    c_bem,
                    Taylor_expansion,
                    w0_NR,
                    0,
                    Tol_w=simulation.omega_tol,
                )
                # print("Iter Total NR I_w=%d: w=%g" % (Iter_w, w))
            else:
                if eps >= simulation.lim_eps:
                    w, Iter_w = w_optimal_by_Newton_Rapson(
                        state,
                        simulation,
                        solute,
                        c_bem,
                        Taylor_expansion,
                        1,
                        0,
                        Tol_w=simulation.omega_tol,
                    )
                    # print("Iter Total NR I_w=%d: w=%g" % (Iter_w, w))

            # Section 5: Calculate the norm and update for next iteration.
            state.d_soln0_nl = state.d_soln_nl * w
            state.soln0_nl = state.soln0_nl + state.d_soln0_nl
            eps = np.linalg.norm(state.d_soln0_nl, ord=np.inf)
            u_nl = dolfinx.fem.Function(solute.fenics_space)
            u_nl.x.array[:] = np.ascontiguousarray(state.soln0_nl[:fem_size])
            u0_nl.x.array[:] = u_nl.x.array[:]
            print("Newton-Raphson iteration=%d: residual=%g" % (Iter, eps))
            simulation.run_info["solver_error_npbe"].append(eps)

            # Section 6: Calculate the norm and update for next iteration.
            while ((eps / (1.25 * gmres_tol)) // 1000 == 0) and (
                gmres_tol
                >= simulation.gmres_tolerance * simulation.gmres_tolerance_factor_npbe
            ):
                gmres_tol = gmres_tol * (1 / simulation.gmres_tolerance_factor_npbe)

            end2 = time.time()
            # Total time to solve 1 nonlinear iteration
            curr_time2 = end2 - start2
            simulation.timings["time_npbe_it"].append(curr_time2)

        iters = len(simulation.run_info["solver_error_npbe"])
        simulation.run_info["solver_iteration_count_npbe_newton"] = iters

        u_T = dolfinx.fem.Function(solute.fenics_space)
        u_T.x.array[:] = u_nl.x.array[:] + u_l.x.array[:]
        solute.results["phi_fem"] = (
            u_T / C1
        )  # using C1 to convert from nondimensional to PyGBe units

        d_phi_s = bempp_cl.api.GridFunction(
            solute.bempp_space,
            coefficients=state.soln0_nl[fem_size:].real,
        )
        phi_s = bempp_cl.api.GridFunction(
            solute.bempp_space,
            coefficients=function_Um(solute.trace_space.grid, u_T, solute.mesh_v).real,
        )
        solute.results["phi_s"] = phi_s / C1
        solute.results["d_phi_s"] = d_phi_s / C1

        Um_T0 = function_Um(solute.mesh, u_T, solute.mesh_v)
        Um_T = bempp_cl.api.GridFunction(solute.bempp_space0, coefficients=Um_T0.real)
        rhs_0_values = rhs_0(solute, Um_T, C1 * solute.results["phi_coul"])

        identity = bempp_cl.api.operators.boundary.sparse.identity(
            solute.bempp_space0, solute.bempp_space0, solute.bempp_space0
        ).weak_form()
        P_0 = InverseSparseDiscreteBoundaryOperator(identity)

        callback = IterationCounter(True)
        start1 = time.time()
        Sol_T, info = gmres(
            solute.matrices["A0"],
            rhs_0_values,
            M=P_0,
            rtol=solute.gmres_tolerance,
            restart=solute.gmres_restart,
            callback=callback,
            callback_type="legacy",
        )
        simulation.timings["time_gmres_npbe_dphi"] = time.time() - start1
        simulation.run_info["solver_iteration_count_npbe_dphi"] = callback.count

        dUm_T = bempp_cl.api.GridFunction(solute.bempp_space0, coefficients=Sol_T.real)
        solute.results["phi"] = (
            Um_T / C1
        )  # using C1 to convert from nondimensional to PyGBe units
        solute.results["d_phi"] = (
            dUm_T / C1
        )  # using C1 to convert from nondimensional to PyGBe units

    return


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


def function_Um(grid, u, mesh):
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


def Taylor_Expansion_of_vector_c(Taylor_expansion, u0_nl, u_l, fenics_space):
    US = u0_nl + u_l
    if Taylor_expansion == "T3":
        # print("Case: Cubic Taylor Exp.")
        expr_ufl_S = US + np.power(US, 3) / 6
        expr_ufl_C = 1 + np.power(US, 2) / 2
    elif Taylor_expansion == "T11":
        # print("Case: Taylor's 11th Exp.")
        expr_ufl_S = (
            US
            + np.power(US, 3) / 6
            + np.power(US, 5) / 120
            + np.power(US, 7) / 5040
            + np.power(US, 9) / 362880
            + np.power(US, 11) / 39916800
        )
        expr_ufl_C = (
            1
            + np.power(US, 2) / 2
            + np.power(US, 4) / 24
            + np.power(US, 6) / 720
            + np.power(US, 8) / 40320
            + np.power(US, 10) / 3628800
        )
    else:
        # print("Case: Hyperbolic Sine")
        expr_ufl_S = ufl.sinh(US)
        expr_ufl_C = ufl.cosh(US)
    NL_Fem_G_S = dolfinx.fem.Function(fenics_space)
    NL_Fem_G_C = dolfinx.fem.Function(fenics_space)
    NL_Fem_G_S.interpolate(
        dolfinx.fem.Expression(expr_ufl_S, fenics_space.element.interpolation_points)
    )
    NL_Fem_G_C.interpolate(
        dolfinx.fem.Expression(expr_ufl_C, fenics_space.element.interpolation_points)
    )
    return NL_Fem_G_S, NL_Fem_G_C


def Scheme_election(state, simulation, solute, c_bem):
    # Calculation of values fd_w1(1).
    fD_w0, _ = Evaluate_fd_dfd(state, simulation, solute, c_bem, "SINH", 1, False)
    # print("iter S=%d: norm=%g: w=%g" % (0, abs(fD_w0), 1))
    # print("Evaluate point w0=%g" % (1))
    # Choice of scheme
    Scheme = "T3-HS"
    if np.abs(fD_w0) < 100000:
        Bisection_Secant_Method = False
    else:
        Bisection_Secant_Method = True
        if math.isnan(fD_w0):  # First overflow correction
            Scheme = "T3-T11"
    # Initial relaxation factor w0_NR for Newton-Raphson
    if np.abs(fD_w0) <= 10:
        w0_NR = 1
    else:
        w0_NR = 2
    # print("Scheme " + Scheme)
    # Important variables that depend on the scheme used
    if Scheme == "T3-HS":
        Taylor_expansion_list = ["SINH"]
    elif Scheme == "T3-T11":
        Taylor_expansion_list = ["T11"]
    Iter_Transition = len(Taylor_expansion_list)
    return (
        Scheme,
        Taylor_expansion_list,
        w0_NR,
        Iter_Transition,
        Bisection_Secant_Method,
    )


def Evaluate_fd_dfd(
    state, simulation, solute, c_bem, taylor_expansion, w0, derivate=False
):

    fem_size = solute.fenics_space.dofmap.index_map.size_global
    v = ufl.TestFunction(solute.fenics_space)
    KI = solute.ep_ex * (solute.kappa**2) * solute.Alpha
    soln0_nl_S = state.soln_l + state.soln0_nl + state.d_soln_nl * w0
    soln0_nl_dS = state.d_soln_nl
    soln_fem_nl_S = soln0_nl_S[:fem_size]
    u_nl_S = dolfinx.fem.Function(solute.fenics_space)
    u_nl_S.x.array[:] = np.ascontiguousarray(soln_fem_nl_S)

    US = u_nl_S
    if taylor_expansion == "T11":
        expr_ufl_S = (
            US
            + US**3 / 6
            + US**5 / 120
            + US**7 / 5040
            + US**9 / 362880
            + US**11 / 39916800
        )
        expr_ufl_dS = (
            1 + US**2 / 2 + US**4 / 24 + US**6 / 720 + US**8 / 40320 + US**10 / 3628800
        )
        if derivate:
            expr_ufl_ddS = US + US**3 / 6 + US**5 / 120 + US**7 / 5040 + US**9 / 362880
    elif taylor_expansion == "SINH":
        expr_ufl_S = ufl.sinh(US)
        expr_ufl_dS = ufl.cosh(US)
    else:
        raise ValueError(f"Taylor expansion no soportada: {taylor_expansion}")

    NL_Fem_G_S = dolfinx.fem.Function(solute.fenics_space)
    NL_Fem_G_S.interpolate(
        dolfinx.fem.Expression(
            expr_ufl_S, solute.fenics_space.element.interpolation_points
        )
    )
    NL_Fem_G_dS = dolfinx.fem.Function(solute.fenics_space)
    NL_Fem_G_dS.interpolate(
        dolfinx.fem.Expression(
            expr_ufl_dS, solute.fenics_space.element.interpolation_points
        )
    )

    c_fem_S = dolfinx.fem.assemble_vector(
        dolfinx.fem.form(KI * NL_Fem_G_S * v * ufl.dx)
    ).array
    c_nlG_S = np.concatenate([c_fem_S, c_bem])
    c_fem_dS = dolfinx.fem.assemble_vector(
        dolfinx.fem.form(KI * NL_Fem_G_dS * v * ufl.dx)
    ).array
    c_nlG_dS = np.concatenate([c_fem_dS, c_bem])

    rhs_A = -(solute.matrices["A_nl"] * soln0_nl_dS)
    rhs_B = -(
        solute.matrices["A_nl"] * (state.soln_l + state.soln0_nl)
        - simulation.rhs["rhs_discrete"]
        + c_nlG_S
    )
    rhs_dB = -(c_nlG_dS * soln0_nl_dS)
    AA_AdB = np.dot(rhs_A + rhs_dB, rhs_A)
    AB_BdB = np.dot(rhs_A + rhs_dB, rhs_B)
    fD_w0 = -AB_BdB / AA_AdB - w0

    if not derivate:
        return fD_w0, None

    if taylor_expansion == "SINH":
        c_nlG_ddS = c_nlG_S
    else:
        NL_Fem_G_ddS = dolfinx.fem.Function(solute.fenics_space)
        NL_Fem_G_ddS.interpolate(
            dolfinx.fem.Expression(
                expr_ufl_ddS, solute.fenics_space.element.interpolation_points
            )
        )
        c_fem_ddS = dolfinx.fem.assemble_vector(
            dolfinx.fem.form(KI * NL_Fem_G_ddS * v * ufl.dx)
        ).array
        c_nlG_ddS = np.concatenate([c_fem_ddS, c_bem])

    rhs_ddB = -(c_nlG_ddS * (soln0_nl_dS**2))
    AddB = np.dot(rhs_A, rhs_ddB)
    BddB = np.dot(rhs_B, rhs_ddB)
    AdB_dBdB = np.dot(rhs_A + rhs_dB, rhs_dB)
    dfD_w0 = (-AA_AdB * (AdB_dBdB + BddB) + AB_BdB * AddB) / (AA_AdB**2) - 1
    return fD_w0, dfD_w0


def w_optimal_by_Bisection(
    state,
    simulation,
    solute,
    c_bem,
    Taylor_expansion,
    wa,
    wb,
    Iter0,
    Secant_equation,
    Tol_w,
):
    Iter = Iter0
    Tol_aditional = 0.001
    eps = 10
    w, d = (wa + wb) / 2, (wb - wa) / 2
    while eps > Tol_w:
        w0, d = w, d / 2
        Iter = Iter + 1
        fD, _ = Evaluate_fd_dfd(
            state, simulation, solute, c_bem, Taylor_expansion, w0, derivate=False
        )
        Sgn, eps = np.sign(fD), abs(fD)
        if Sgn > 0:
            w = w0 + d
        else:
            w = w0 - d
        # print("Iter BI S=%d: norm=%g: w=%g" % (Iter, eps, w0))
        if Secant_equation:
            if Iter == (1 + Iter0):
                wb_sec, fD_wb_sec = w0, fD
                wc_sec = 100
            else:
                wa_sec, fD_wa_sec = wb_sec, fD_wb_sec
                wb_sec, fD_wb_sec = w0, fD
                wc0_sec = wc_sec
                wc_sec = wa_sec - (wa_sec - wb_sec) * (fD_wa_sec) / (
                    fD_wa_sec - fD_wb_sec
                )  # Equation of the line
                if Iter == (2 + Iter0):
                    # print(
                    #     "w calculate for Secant equation use point S=%d: and S=%d: wc_sec=%g:"
                    #     % (Iter, (Iter - 1), wc_sec)
                    # )
                    return
                else:
                    Diff = abs(wc_sec - wc0_sec)
                    # print(
                    #     "w calculate for Secant equation use point S=%d: and S=%d: wc_sec=%g: Diff=%g"
                    #     % (Iter, (Iter - 1), wc_sec, Diff)
                    # )
                    if Diff < 0.05 and abs(fD_wb_sec) < 50 and abs(fD_wa_sec) < 50:
                        break
        if Iter == 50 or w0 >= (wb - Tol_aditional) or w0 <= (wa + Tol_aditional):
            break
    return w0, Iter, wc_sec


def w_optimal_by_Newton_Rapson(
    state, simulation, solute, c_bem, Taylor_expansion, w0, Iter0, Tol_w
):
    Iter = Iter0
    fD, dfD = Evaluate_fd_dfd(
        state, simulation, solute, c_bem, Taylor_expansion, w0, derivate=True
    )
    w, eps = w0 - fD / dfD, abs(fD)
    # print("Iter NR S=%d: norm=%g: w=%g" % (Iter, eps, w0))

    while eps > Tol_w:
        w0 = w
        Iter = Iter + 1
        fD, dfD = Evaluate_fd_dfd(
            state, simulation, solute, c_bem, Taylor_expansion, w0, derivate=True
        )
        w, eps = w0 - fD / dfD, abs(fD)
        # print("Iter NR S=%d: norm=%g: w=%g" % (Iter, eps, w0))

        if Iter == 30:
            break
    return w0, Iter
