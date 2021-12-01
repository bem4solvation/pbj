import numpy as np
import bempp.api
import os
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

invert_potential = True


def verify_parameters(self):
    return True


def lhs(self):
    dirichl_space = self.dirichl_space
    neumann_space = self.neumann_space
    ep_in = self.ep_in
    ep_ex = self.ep_ex
    kappa = self.kappa
    operator_assembler = self.operator_assembler

    dlp_in = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    slp_in = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    hlp_in = laplace.hypersingular(dirichl_space, neumann_space, neumann_space, assembler=operator_assembler)
    adlp_in = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space, assembler=operator_assembler)

    dlp_ex = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa,
                                             assembler=operator_assembler)
    slp_ex = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                             assembler=operator_assembler)
    hlp_ex = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa,
                                              assembler=operator_assembler)
    adlp_ex = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa,
                                                      assembler=operator_assembler)

    phi_identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_identity = sparse.identity(neumann_space, neumann_space, neumann_space)

    ep = ep_ex / ep_in

    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (0.5*(1+(1.0/ep)))*phi_identity - dlp_ex + (1.0/ep)*dlp_in
    A[0, 1] = slp_ex - slp_in
    A[1, 0] = (1.0/ep)*hlp_ex - (1.0/ep)*hlp_in
    A[1, 1] = (0.5*(1+(1.0/ep)))*dph_identity + (1.0/ep)*adlp_ex - adlp_in

    self.matrices["A"] = A


def rhs(self):
    dirichl_space = self.dirichl_space
    neumann_space = self.neumann_space
    q = self.q
    x_q = self.x_q
    ep_ex = self.ep_ex
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
            result[:] = values[:,0] / ep_ex

        @bempp.api.callable(vectorized=True)
        def rhs2_fun(x, n, domain_index, result):
            import exafmm.laplace as _laplace
            sources = _laplace.init_sources(x_q, q)
            targets = _laplace.init_targets(x.T)
            fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename='.rhs.tmp')
            tree = _laplace.setup(sources, targets, fmm)
            values = _laplace.evaluate(tree, fmm)
            os.remove('.rhs.tmp')
            result[:] = np.sum(values[:,1:] * n.T, axis=1) / ep_ex

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=rhs1_fun)
        rhs_2 = bempp.api.GridFunction(neumann_space, fun=rhs2_fun)

    else:
        @bempp.api.real_callable
        def d_green_func(x, n, domain_index, result):
            nrm = np.sqrt((x[0]-x_q[:, 0])**2 + (x[1]-x_q[:, 1])**2 + (x[2]-x_q[:, 2])**2)
            const = -1./(4.*np.pi*ep_ex)
            result[:] = const*np.sum(q*np.dot(x-x_q, n)/(nrm**3))

        @bempp.api.real_callable
        def green_func(x, n, domain_index, result):
            nrm = np.sqrt((x[0]-x_q[:, 0])**2 + (x[1]-x_q[:, 1])**2 + (x[2]-x_q[:, 2])**2)
            result[:] = np.sum(q/nrm)/(4.*np.pi*ep_ex)

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=green_func)
        rhs_2 = bempp.api.GridFunction(dirichl_space, fun=d_green_func)

    self.rhs["rhs_1"], self.rhs["rhs_2"] = rhs_1, rhs_2


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
