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
    ep_out = self.ep_ex
    kappa = self.kappa
    operator_assembler = self.operator_assembler
    
    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    slp_in = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    dlp_in = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler)
    slp_out = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                              assembler=operator_assembler)
    dlp_out = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa,
                                              assembler=operator_assembler)

    A = bempp.api.BlockedOperator(2, 2)

    A[0, 0] = 0.5 * identity + dlp_in
    A[0, 1] = -(ep_out / ep_in) * slp_in
    A[1, 0] = 0.5 * identity - dlp_out
    A[1, 1] = slp_out

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

    self.rhs["rhs_1"], self.rhs["rhs_2"] = rhs_1, rhs_2
