import numpy as np
import os
import bempp.api
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

invert_potential = False


def verify_parameters(self):
    alpha = self.pb_formulation_alpha
    beta = self.pb_formulation_beta
    if np.isnan(alpha) or np.isnan(beta):
        raise ValueError("pb_formulation_alpha and pb_formulation_beta not defined in Solute class")
    return True


def lhs(self):
    dirichl_space = self.dirichl_space
    neumann_space = self.neumann_space
    ep_in = self.ep_in
    ep_ex = self.ep_ex
    kappa = self.kappa
    alpha = self.pb_formulation_alpha
    beta = self.pb_formulation_beta
    operator_assembler = self.operator_assembler

    phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_id = sparse.identity(neumann_space, neumann_space, neumann_space)

    ep = ep_ex / ep_in

    A_in = laplace_multitrace(dirichl_space, neumann_space, operator_assembler)
    A_ex = mod_helm_multitrace(dirichl_space, neumann_space, kappa, operator_assembler)

    D = bempp.api.BlockedOperator(2, 2)
    D[0, 0] = alpha * phi_id
    D[0, 1] = 0.0 * phi_id
    D[1, 0] = 0.0 * phi_id
    D[1, 1] = beta * dph_id

    E = bempp.api.BlockedOperator(2, 2)
    E[0, 0] = phi_id
    E[0, 1] = 0.0 * phi_id
    E[1, 0] = 0.0 * phi_id
    E[1, 1] = dph_id * (1.0 / ep)

    F = bempp.api.BlockedOperator(2, 2)
    F[0, 0] = alpha * phi_id
    F[0, 1] = 0.0 * phi_id
    F[1, 0] = 0.0 * phi_id
    F[1, 1] = dph_id * (beta / ep)

    Id = bempp.api.BlockedOperator(2, 2)
    Id[0, 0] = phi_id
    Id[0, 1] = 0.0 * phi_id
    Id[1, 0] = 0.0 * phi_id
    Id[1, 1] = dph_id

    interior_projector = ((0.5 * Id) + A_in)
    scaled_exterior_projector = (D * ((0.5 * Id) - A_ex) * E)
    A = ((0.5 * Id) + A_in) + (D * ((0.5 * Id) - A_ex) * E) - (Id + F)
    
    self.matrices["A"] = A
    self.matrices["A_in"] = A_in
    self.matrices["A_ex"] = A_ex
    self.matrices["interior_projector"] = interior_projector
    self.matrices["scaled_exterior_projector"] = scaled_exterior_projector


def rhs(self):
    dirichl_space = self.dirichl_space
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
            fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename='.rhs.tmp')
            tree = _laplace.setup(sources, targets, fmm)
            values = _laplace.evaluate(tree, fmm)
            os.remove('.rhs.tmp')
            result[:] = (-1.0)*values[:, 0] / ep_in

        @bempp.api.callable(vectorized=True)
        def fmm_d_green_func(x, n, domain_index, result):
            import exafmm.laplace as _laplace
            sources = _laplace.init_sources(x_q, q)
            targets = _laplace.init_targets(x.T)
            fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename='.rhs.tmp')
            tree = _laplace.setup(sources, targets, fmm)
            values = _laplace.evaluate(tree, fmm)
            os.remove('.rhs.tmp')
            result[:] = (-1.0)*np.sum(values[:, 1:] * n.T, axis=1) / ep_in

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=fmm_green_func)
        rhs_2 = bempp.api.GridFunction(dirichl_space, fun=fmm_d_green_func)

    else:
        @bempp.api.real_callable
        def d_green_func(x, n, domain_index, result):
            nrm = np.sqrt((x[0]-x_q[:, 0])**2 + (x[1]-x_q[:, 1])**2 + (x[2]-x_q[:, 2])**2)
            const = -1./(4.*np.pi*ep_in)
            result[:] = (-1.0)*const*np.sum(q*np.dot(x-x_q, n)/(nrm**3))

        @bempp.api.real_callable
        def green_func(x, n, domain_index, result):
            nrm = np.sqrt((x[0]-x_q[:, 0])**2 + (x[1]-x_q[:, 1])**2 + (x[2]-x_q[:, 2])**2)
            result[:] = (-1.0)*np.sum(q/nrm)/(4.*np.pi*ep_in)

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=green_func)
        rhs_2 = bempp.api.GridFunction(dirichl_space, fun=d_green_func)

    self.rhs["rhs_1"], self.rhs["rhs_2"] = rhs_1, rhs_2


def laplace_multitrace(dirichl_space, neumann_space, operator_assembler):
    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-1.0) * laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    A[0, 1] = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    A[1, 0] = laplace.hypersingular(dirichl_space, neumann_space, neumann_space, assembler=operator_assembler)
    A[1, 1] = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space, assembler=operator_assembler)

    return A


def mod_helm_multitrace(dirichl_space, neumann_space, kappa, operator_assembler):
    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-1.0) * modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa,
                                                       assembler=operator_assembler)
    A[0, 1] = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                              assembler=operator_assembler)
    A[1, 0] = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa,
                                               assembler=operator_assembler)
    A[1, 1] = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa,
                                                      assembler=operator_assembler)

    return A


def block_diagonal_precon_alpha_beta(solute):
    from scipy.sparse import diags, bmat
    from scipy.sparse.linalg import factorized, LinearOperator
    from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
    
    dirichl_space = solute.dirichl_space
    neumann_space = solute.neumann_space
    ep_in = solute.ep_in   
    ep_ex = solute.ep_ex   
    kappa = solute.kappa
    alpha = solute.alpha
    beta = solute.beta

    slp_in_diag = laplace.single_layer(neumann_space, dirichl_space, dirichl_space,
                                       assembler="only_diagonal_part").weak_form().A
    dlp_in_diag = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space,
                                       assembler="only_diagonal_part").weak_form().A
    hlp_in_diag = laplace.hypersingular(dirichl_space, neumann_space, neumann_space,
                                        assembler="only_diagonal_part").weak_form().A
    adlp_in_diag = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space,
                                                assembler="only_diagonal_part").weak_form().A
    
    slp_out_diag = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                                   assembler="only_diagonal_part").weak_form().A
    dlp_out_diag = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa,
                                                   assembler="only_diagonal_part").weak_form().A
    hlp_out_diag = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa,
                                                    assembler="only_diagonal_part").weak_form().A
    adlp_out_diag = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa,
                                                            assembler="only_diagonal_part").weak_form().A

    phi_identity_diag = sparse.identity(dirichl_space, dirichl_space, dirichl_space).weak_form().A.diagonal()
    dph_identity_diag = sparse.identity(neumann_space, neumann_space, neumann_space).weak_form().A.diagonal()

    ep = ep_ex/ep_in
    
    diag11 = diags((-0.5*(1+alpha))*phi_identity_diag + (alpha*dlp_out_diag) - dlp_in_diag)
    diag12 = diags(slp_in_diag - ((alpha/ep)*slp_out_diag))
    diag21 = diags(hlp_in_diag - (beta*hlp_out_diag))
    diag22 = diags((-0.5*(1+(beta/ep)))*dph_identity_diag + adlp_in_diag - ((beta/ep)*adlp_out_diag))
    block_mat_precond = bmat([[diag11, diag12], [diag21, diag22]]).tocsr()  # csr_matrix

    solve = factorized(block_mat_precond)  # a callable for solving a sparse linear system (treat it as an inverse)
    precond = LinearOperator(matvec=solve, dtype='float64', shape=block_mat_precond.shape)
    
    solute.matrices["preconditioning_matrix_gmres"] = aslinearoperator(block_mat_precond)
    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]

    solute.matrices["A_discrete"] = matrix_to_discrete_form(solute.matrices["A_final"], "weak")
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(solute.rhs["rhs_final"], "weak", solute.matrices["A"])


def mass_matrix_preconditioner(solute):
    #Opción A:
    from bempp.api.utils.helpers import get_inverse_mass_matrix
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
    from pbj.electrostatics.solute import matrix_to_discrete_form, rhs_to_discrete_form

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
    #Opción B:
    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]
    solute.matrices["A_discrete"] = matrix_to_discrete_form(solute.matrices["A_final"], "strong")
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(solute.rhs["rhs_final"], "strong", solute.matrices["A"])
    """