import numpy as np
import bempp.api
import os
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

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
    A[0, 0] = phi_identity + dlp_in - dlp_ex
    A[0, 1] = -slp_in + ((1.0/ep) * slp_ex)
    A[1, 0] = -hlp_in + (ep * hlp_ex)
    A[1, 1] = dph_identity - adlp_in + adlp_ex

    self.matrices["A"] = A


def rhs(self):
    dirichl_space = self.dirichl_space
    q = self.q
    x_q = self.x_q
    ep_in = self.ep_in

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

    self.rhs["rhs_1"] = rhs_1
    self.rhs["rhs_2"] = rhs_2