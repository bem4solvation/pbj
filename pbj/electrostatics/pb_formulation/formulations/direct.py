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

    A[0, 0] = 0.5 * identity + dlp_in
    A[0, 1] = -slp_in
    A[1, 0] = 0.5 * identity - dlp_out
    A[1, 1] = (ep_in / ep_out) * slp_out

    self.matrices["A"] = A


def rhs(self):
    dirichl_space = self.dirichl_space
    neumann_space = self.neumann_space
    q = self.q
    x_q = self.x_q
    d = self.d
    Q = self.Q
    ep_in = self.ep_in
    rhs_constructor = self.rhs_constructor
    force_field = self.force_field

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
        def multipolar_charges_fun(x, n, i, result): # not using fmm
            T2 = np.zeros((len(x_q),3,3))
            phi = 0
            dist = x - x_q
            norm = np.sqrt(np.sum((dist*dist), axis = 1))
            T0 = 1/norm[:]
            T1 = np.transpose(dist.transpose()/norm**3)
            T2[:,:,:] = np.ones((len(x_q),3,3))[:]* dist.reshape((len(x_q),1,3))* \
            np.transpose(np.ones((len(x_q),3,3))*dist.reshape((len(x_q),1,3)), (0,2,1))/norm.reshape((len(x_q),1,1))**5
            phi = np.sum(q[:]*T0[:]) + np.sum(T1[:]*d[:]) + 0.5*np.sum(np.sum(T2[:]*Q[:],axis=1))
            result[0] = (phi/(4*np.pi*ep_in))

        # @bempp.api.real_callable
        # def zero(x, n, domain_index, result):
        #     result[0] = 0

        coefs = np.zeros(neumann_space.global_dof_count)

        if force_field == "amoeba":
            rhs_1 = bempp.api.GridFunction(dirichl_space, fun=multipolar_charges_fun)
        else:
            rhs_1 = bempp.api.GridFunction(dirichl_space, fun=fmm_green_func)

        # rhs_2 = bempp.api.GridFunction(neumann_space, fun=zero)
        rhs_2 = bempp.api.GridFunction(neumann_space, coefficients=coefs)

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

        @bempp.api.real_callable
        def multipolar_charges_fun(x, n, i, result):
            T2 = np.zeros((len(x_q),3,3))
            phi = 0
            dist = x - x_q
            norm = np.sqrt(np.sum((dist*dist), axis = 1))
            T0 = 1/norm[:]
            T1 = np.transpose(dist.transpose()/norm**3)
            T2[:,:,:] = np.ones((len(x_q),3,3))[:]* dist.reshape((len(x_q),1,3))* \
            np.transpose(np.ones((len(x_q),3,3))*dist.reshape((len(x_q),1,3)), (0,2,1))/norm.reshape((len(x_q),1,1))**5
            phi = np.sum(q[:]*T0[:]) + np.sum(T1[:]*d[:]) + 0.5*np.sum(np.sum(T2[:]*Q[:],axis=1))
            result[0] = (phi/(4*np.pi*ep_in))
    
        if force_field == "amoeba":
            rhs_1 = bempp.api.GridFunction(dirichl_space, fun=multipolar_charges_fun)
        else:
            rhs_1 = bempp.api.GridFunction(dirichl_space, fun=charges_fun)

        rhs_2 = bempp.api.GridFunction(neumann_space, fun=zero)

    self.rhs["rhs_1"], self.rhs["rhs_2"] = rhs_1, rhs_2


def block_diagonal_preconditioner(solute):
    from scipy.sparse import diags, bmat
    from scipy.sparse.linalg import aslinearoperator
    from pbj.electrostatics.solute import matrix_to_discrete_form, rhs_to_discrete_form

    matrix_A = solute.matrices["A"]

    block1 = matrix_A[0, 0]
    block2 = matrix_A[0, 1]
    block3 = matrix_A[1, 0]
    block4 = matrix_A[1, 1]

    diag11 = (
        block1._op1._alpha * block1._op1._op.weak_form().to_sparse().diagonal()
        + block1._op2.descriptor.singular_part.weak_form().to_sparse().diagonal()
    )
    diag12 = (
        block2._alpha
        * block2._op.descriptor.singular_part.weak_form().to_sparse().diagonal()
    )
    diag21 = (
        block3._op1._alpha * block3._op1._op.weak_form().to_sparse().diagonal()
        + block3._op2._alpha
        * block3._op2._op.descriptor.singular_part.weak_form().to_sparse().diagonal()
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

    #if permuted_rows:
    diag11 = .5 * identity_diag - dlp_out_diag
    diag12 = (ep_in / ep_ex) * slp_out_diag
    diag21 = .5 * identity_diag + dlp_in_diag
    diag22 = -slp_in_diag
    """


def mass_matrix_preconditioner(solute):
    from pbj.electrostatics.solute import matrix_to_discrete_form, rhs_to_discrete_form

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


def calculate_potential(self, rerun_all):
    calculate_potential_one_surface(self, rerun_all)


def lhs_inter_solute_interactions(self, solute_target, solute_source):

   
    dirichl_space_target = solute_target.dirichl_space
    neumann_space_target = solute_target.neumann_space
    dirichl_space_source = solute_source.dirichl_space
    neumann_space_source = solute_source.neumann_space

    ep_in = solute_source.ep_in
    ep_out = self.ep_ex
    kappa = self.kappa
    operator_assembler = self.operator_assembler



    dlp = modified_helmholtz.double_layer(
        dirichl_space_source, dirichl_space_target, dirichl_space_target, kappa, assembler=operator_assembler
    )
    slp = modified_helmholtz.single_layer(
        neumann_space_source, neumann_space_target, neumann_space_target, kappa,  assembler=operator_assembler
    )

    zero_00 = bempp.api.assembly.boundary_operator.ZeroBoundaryOperator(
        dirichl_space_source, dirichl_space_target, dirichl_space_target
    )
    
    zero_01 = bempp.api.assembly.boundary_operator.ZeroBoundaryOperator(
        neumann_space_source, neumann_space_target, neumann_space_target
    )
    
    A_inter = bempp.api.BlockedOperator(2, 2)

    
    A_inter[0, 0] = zero_00
    A_inter[0, 1] = zero_01 
    A_inter[1, 0] = - dlp
    A_inter[1, 1] = (ep_in / ep_out) * slp

    
    return A_inter
