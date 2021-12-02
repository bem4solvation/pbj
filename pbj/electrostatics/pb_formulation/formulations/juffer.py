import numpy as np
import bempp.api
import os
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

invert_potential = False


def verify_parameters(self):
    return True


def lhs(self):
    dirichl_space = self.dirichl_space
    neumann_space = self.neumann_space
    ep_in = self.ep_in
    ep_ex = self.ep_ex
    kappa = self.kappa
    operator_assembler = self.operator_assembler

    phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_id = sparse.identity(neumann_space, neumann_space, neumann_space)
    ep = ep_ex / ep_in

    dF = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    dP = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa,
                                         assembler=operator_assembler)
    L1 = (ep * dP) - dF

    F = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    P = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                        assembler=operator_assembler)
    L2 = F - P

    ddF = laplace.hypersingular(dirichl_space, neumann_space, neumann_space, assembler=operator_assembler)
    ddP = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa,
                                           assembler=operator_assembler)
    L3 = ddP - ddF

    dF0 = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space, assembler=operator_assembler)
    dP0 = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa,
                                                  assembler=operator_assembler)
    L4 = dF0 - ((1.0 / ep) * dP0)

    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (0.5 * (1.0 + ep) * phi_id) - L1
    A[0, 1] = (-1.0) * L2
    A[1, 0] = L3  # Sign change due to bempp definition
    A[1, 1] = (0.5 * (1.0 + (1.0 / ep)) * dph_id) - L4

    self.matrices["A"] = A


def rhs(self):
    dirichl_space = self.dirichl_space
    neumann_space = self.neumann_space
    q = self.q
    x_q = self.x_q
    ep_in = self.ep_in
    rhs_constructor = self.rhs_constructor

    if rhs_constructor == "fmm":
        @bempp.api.callable(vectorized=True)
        def rhs1_fun(x, n, domain_index, result):
            import exafmm.laplace as _laplace
            sources = _laplace.init_sources(x_q, q)
            targets = _laplace.init_targets(x.T)
            fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename='.rhs.tmp')
            tree = _laplace.setup(sources, targets, fmm)
            values = _laplace.evaluate(tree, fmm)
            os.remove('.rhs.tmp')
            result[:] = values[:,0] / ep_in

        @bempp.api.callable(vectorized=True)
        def rhs2_fun(x, n, domain_index, result):
            import exafmm.laplace as _laplace
            sources = _laplace.init_sources(x_q, q)
            targets = _laplace.init_targets(x.T)
            fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename='.rhs.tmp')
            tree = _laplace.setup(sources, targets, fmm)
            values = _laplace.evaluate(tree, fmm)
            os.remove('.rhs.tmp')
            result[:] = np.sum(values[:,1:] * n.T, axis=1) / ep_in

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=rhs1_fun)
        rhs_2 = bempp.api.GridFunction(neumann_space, fun=rhs2_fun)

    else:
        @bempp.api.real_callable
        def d_green_func(x, n, domain_index, result):
            nrm = np.sqrt((x[0]-x_q[:, 0])**2 + (x[1]-x_q[:, 1])**2 + (x[2]-x_q[:, 2])**2)
            const = -1./(4.*np.pi*ep_in)
            result[:] = const*np.sum(q*np.dot(x-x_q, n)/(nrm**3))

        @bempp.api.real_callable
        def green_func(x, n, domain_index, result):
            nrm = np.sqrt((x[0]-x_q[:, 0])**2 + (x[1]-x_q[:, 1])**2 + (x[2]-x_q[:, 2])**2)
            result[:] = np.sum(q/nrm)/(4.*np.pi*ep_in)

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=green_func)
        rhs_2 = bempp.api.GridFunction(dirichl_space, fun=d_green_func)

    self.rhs["rhs_1"], self.rhs["rhs_2"] = rhs_1, rhs_2


def block_diagonal_preconditioner(solute):
    from scipy.sparse import diags, bmat
    from scipy.sparse.linalg import factorized, LinearOperator
    from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

    dirichl_space = solute.dirichl_space
    neumann_space = solute.neumann_space
    ep_in = solute.ep_in
    ep_ex = solute.ep_ex
    kappa = solute.kappa

    phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space).weak_form().A.diagonal()
    dph_id = sparse.identity(neumann_space, neumann_space, neumann_space).weak_form().A.diagonal()
    ep = ep_ex/ep_in

    dF = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space,
                              assembler="only_diagonal_part").weak_form().A
    dP = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa,
                                         assembler="only_diagonal_part").weak_form().A
    L1 = (ep*dP) - dF

    F = laplace.single_layer(neumann_space, dirichl_space, dirichl_space,
                             assembler="only_diagonal_part").weak_form().A
    P = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                        assembler="only_diagonal_part").weak_form().A
    L2 = F - P

    ddF = laplace.hypersingular(dirichl_space, neumann_space, neumann_space,
                                assembler="only_diagonal_part").weak_form().A
    ddP = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa,
                                           assembler="only_diagonal_part").weak_form().A
    L3 = ddP - ddF

    dF0 = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space,
                                       assembler="only_diagonal_part").weak_form().A
    dP0 = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa,
                                                  assembler="only_diagonal_part").weak_form().A
    L4 = dF0 - ((1.0/ep)*dP0)

    diag11 = diags((0.5*(1.0 + ep)*phi_id) - L1)
    diag12 = diags((-1.0)*L2)
    diag21 = diags(L3)
    diag22 = diags((0.5*(1.0 + (1.0/ep))*dph_id) - L4)
    block_mat_precond = bmat([[diag11, diag12], [diag21, diag22]]).tocsr()  # csr_matrix

    #Error:
    #diag11 = diags((0.5*(1.0 + ep)*phi_id) - L1)
    #File "D:\ProgramData\Anaconda3\envs\pbj_env\lib\site-packages\scipy\sparse\construct.py", line 141, in diags
    #raise ValueError("Different number of diagonals and offsets.")
    #ValueError: Different number of diagonals and offsets.

    solve = factorized(block_mat_precond)  # a callable for solving a sparse linear system (treat it as an inverse)
    precond = LinearOperator(matvec=solve, dtype='float64', shape=block_mat_precond.shape)


    solute.matrices["preconditioning_matrix_gmres"] = precond
    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]

    solute.matrices["A_discrete"] = matrix_to_discrete_form(solute.matrices["A_final"], "weak")
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(solute.rhs["rhs_final"], "weak", solute.matrices["A"])


def mass_matrix_preconditioner(solute):
    from pbj.electrostatics.solute import matrix_to_discrete_form, rhs_to_discrete_form
    #Option A:
    """
    from bempp.api.utils.helpers import get_inverse_mass_matrix
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

    matrix = solute.matrices["A"]
    nrows = len(matrix.range_spaces)
    range_ops = np.empty((nrows, nrows), dtype="O")

    for index in range(nrows):
        range_ops[index, index] = get_inverse_mass_matrix(matrix.range_spaces[index],
                                                          matrix.dual_to_range_spaces[index])

    preconditioner = BlockedDiscreteOperator(range_ops)
    solute.matrices['preconditioning_matrix_gmres'] = preconditioner
    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]
    solute.matrices["A_discrete"] = matrix_to_discrete_form(solute.matrices["A_final"], "weak")
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(solute.rhs["rhs_final"], "weak", solute.matrices["A"])

    """
    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]
    solute.matrices["A_discrete"] = matrix_to_discrete_form(solute.matrices["A_final"], "strong")
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(solute.rhs["rhs_final"], "strong", solute.matrices["A"])


def scaled_mass_preconditioner(solute):
    from bempp.api.utils.helpers import get_inverse_mass_matrix
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
    from pbj.electrostatics.solute import matrix_to_discrete_form, rhs_to_discrete_form

    ep_in = solute.ep_in
    ep_ex = solute.ep_ex
    matrix = solute.matrices["A"]

    nrows = len(matrix.range_spaces)
    range_ops = np.empty((nrows, nrows), dtype="O")

    for index in range(nrows):
        range_ops[index, index] = get_inverse_mass_matrix(matrix.range_spaces[index],
                                                          matrix.dual_to_range_spaces[index])

    range_ops[0, 0] = range_ops[0, 0] * (1.0 / (0.5 * (1.0 + (ep_ex/ep_in))))
    range_ops[1, 1] = range_ops[1, 1] * (1.0 / (0.5*(1.0+(ep_in/ep_ex))))

    preconditioner = BlockedDiscreteOperator(range_ops)
    solute.matrices['preconditioning_matrix_gmres'] = preconditioner
    #solute.matrices['preconditioning_matrix'] = preconditioner
    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]
    #multiplicar por precondicionador
    solute.matrices["A_discrete"] = matrix_to_discrete_form(solute.matrices["A_final"], "weak")
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(solute.rhs["rhs_final"], "weak", solute.matrices["A"])



    