import numpy as np
import bempp.api

from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

def lhs(dirichl_space, neumann_space, ep_in, ep_out, kappa, operator_assembler, permute_rows = False):
    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    slp_in = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    dlp_in = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    slp_out = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                              assembler=operator_assembler)
    dlp_out = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa,
                                              assembler=operator_assembler)

    A = bempp.api.BlockedOperator(2, 2)
    if permute_rows:    #Use permuted rows formulation
        A[0, 0] = 0.5 * identity - dlp_out
        A[0, 1] = (ep_in / ep_out) * slp_out
        A[1, 0] = 0.5 * identity + dlp_in
        A[1, 1] = -slp_in
    else:   #Normal direct formulation
        A[0, 0] = 0.5 * identity + dlp_in
        A[0, 1] = -slp_in
        A[1, 0] = 0.5 * identity - dlp_out
        A[1, 1] = (ep_in / ep_out) * slp_out

    return A


import os


def rhs(dirichl_space, neumann_space, q, x_q, ep_in, rhs_constructor):
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
            result[:] = values[:, 0] / ep_in

        # @bempp.api.real_callable
        # def zero(x, n, domain_index, result):
        #     result[0] = 0

        coefs = np.zeros(neumann_space.global_dof_count)

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=fmm_green_func)
        # rhs_2 = bempp.api.GridFunction(neumann_space, fun=zero)
        rhs_2 = bempp.api.GridFunction(neumann_space, coefficients=coefs)

    else:
        @bempp.api.real_callable
        def charges_fun(x, n, domain_index, result):
            nrm = np.sqrt((x[0] - x_q[:, 0]) ** 2 + (x[1] - x_q[:, 1]) ** 2 + (x[2] - x_q[:, 2]) ** 2)
            aux = np.sum(q / nrm)
            result[0] = aux / (4 * np.pi * ep_in)

        @bempp.api.real_callable
        def zero(x, n, domain_index, result):
            result[0] = 0

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=charges_fun)
        rhs_2 = bempp.api.GridFunction(neumann_space, fun=zero)

    return rhs_1, rhs_2
