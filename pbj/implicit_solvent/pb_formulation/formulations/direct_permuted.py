import numpy as np
import bempp.api
import os
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
from .common import calculate_potential_one_surface

invert_potential = False


def verify_parameters(self):
    return True


def lhs(self):
    dirichl_space = self.dirichl_space
    neumann_space = self.neumann_space
    ep_in = self.ep_in
    ep_out = self.ep_ex
    kappa = self.kappa
    operator_assembler = self.operator_assembler

    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    slp_in = laplace.single_layer(
        neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler
    )
    dlp_in = laplace.double_layer(
        dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler
    )
    slp_out = modified_helmholtz.single_layer(
        neumann_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler
    )
    dlp_out = modified_helmholtz.double_layer(
        dirichl_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler
    )

    A = bempp.api.BlockedOperator(2, 2)

    A[0, 0] = 0.5 * identity - dlp_out
    A[0, 1] = (ep_in / ep_out) * slp_out
    A[1, 0] = 0.5 * identity + dlp_in
    A[1, 1] = -slp_in

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
        def fmm_green_func(x, n, domain_index, result):
            import exafmm.laplace as _laplace

            sources = _laplace.init_sources(x_q, q)
            targets = _laplace.init_targets(x.T)
            fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename=".rhs.tmp")
            tree = _laplace.setup(sources, targets, fmm)
            values = _laplace.evaluate(tree, fmm)
            os.remove(".rhs.tmp")
            result[:] = values[:, 0] / ep_in

        @bempp.api.real_callable
        def zero(x, n, domain_index, result):
            result[0] = 0

        rhs_1 = bempp.api.GridFunction(neumann_space, fun=zero)
        rhs_2 = bempp.api.GridFunction(dirichl_space, fun=fmm_green_func)

    else:

        @bempp.api.real_callable
        def charges_fun(x, n, domain_index, result):
            nrm = np.sqrt(
                (x[0] - x_q[:, 0]) ** 2
                + (x[1] - x_q[:, 1]) ** 2
                + (x[2] - x_q[:, 2]) ** 2
            )
            aux = np.sum(q / nrm)
            result[0] = aux / (4 * np.pi * ep_in)

        @bempp.api.real_callable
        def zero(x, n, domain_index, result):
            result[0] = 0

        rhs_1 = bempp.api.GridFunction(neumann_space, fun=zero)
        rhs_2 = bempp.api.GridFunction(dirichl_space, fun=charges_fun)

    self.rhs["rhs_1"], self.rhs["rhs_2"] = rhs_1, rhs_2


def block_diagonal_preconditioner(solute):
    from scipy.sparse import diags, bmat
    from scipy.sparse.linalg import aslinearoperator
    from pbj.implicit_solvent.utils import matrix_to_discrete_form, rhs_to_discrete_form

    matrix_A = solute.matrices["A"]

    block1 = matrix_A[0, 0]
    block2 = matrix_A[0, 1]
    block3 = matrix_A[1, 0]
    block4 = matrix_A[1, 1]

    diag11 = (
        block1._op1._alpha * block1._op1._op.weak_form().to_sparse().diagonal()
        + block1._op2._alpha
        * block1._op2._op.descriptor.singular_part.weak_form().to_sparse().diagonal()
    )
    diag12 = (
        block2._alpha
        * block2._op.descriptor.singular_part.weak_form().to_sparse().diagonal()
    )
    diag21 = (
        block3._op1._alpha * block3._op1._op.weak_form().to_sparse().diagonal()
        + block3._op2.descriptor.singular_part.weak_form().to_sparse().diagonal()
    )
    diag22 = (
        block4._alpha
        * block4._op.descriptor.singular_part.weak_form().to_sparse().diagonal()
    )

    d_aux = 1 / (diag22 - diag21 * diag12 / diag11)
    diag11_inv = 1 / diag11 + 1 / diag11 * diag12 * d_aux * diag21 / diag11
    diag12_inv = -1 / diag11 * diag12 * d_aux
    diag21_inv = -d_aux * diag21 / diag11
    diag22_inv = d_aux

    block_mat_precond = bmat(
        [[diags(diag11_inv), diags(diag12_inv)], [diags(diag21_inv), diags(diag22_inv)]]
    ).tocsr()

    solute.matrices["preconditioning_matrix_gmres"] = aslinearoperator(
        block_mat_precond
    )
    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]

    solute.matrices["A_discrete"] = matrix_to_discrete_form(
        solute.matrices["A_final"], "weak"
    )
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(
        solute.rhs["rhs_final"], "weak", solute.matrices["A"]
    )

    """
    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    identity_diag = identity.weak_form().to_sparse().diagonal()
    slp_in_diag = laplace.single_layer(neumann_space, dirichl_space, dirichl_space,
                                       assembler="only_diagonal_part").weak_form().get_diagonal()
    dlp_in_diag = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space,
                                       assembler="only_diagonal_part").weak_form().get_diagonal()
    slp_out_diag = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                                   assembler="only_diagonal_part").weak_form().get_diagonal()
    dlp_out_diag = modified_helmholtz.double_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                                   assembler="only_diagonal_part").weak_form().get_diagonal()
    """


def mass_matrix_preconditioner(solute):
    from pbj.implicit_solvent.utils import matrix_to_discrete_form, rhs_to_discrete_form

    # Option A:
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
    solute.matrices["A_discrete"] = matrix_to_discrete_form(
        solute.matrices["A_final"], "strong"
    )
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(
        solute.rhs["rhs_final"], "strong", solute.matrices["A"]
    )


def calculate_potential(self, rerun_all, rerun_rhs):
    calculate_potential_one_surface(self, rerun_all, rerun_rhs)
