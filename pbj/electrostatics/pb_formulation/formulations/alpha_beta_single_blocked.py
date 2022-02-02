import numpy as np
import bempp.api
import os
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
from .common import calculate_potential_one_surface


invert_potential = False


def verify_parameters(self):
    alpha = self.pb_formulation_alpha
    beta = self.pb_formulation_beta
    if np.isnan(alpha) or np.isnan(beta):
        raise ValueError(
            "pb_formulation_alpha and pb_formulation_beta not defined in Solute class"
        )
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

    dlp_in = laplace.double_layer(
        dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler
    )
    slp_in = laplace.single_layer(
        neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler
    )
    hlp_in = laplace.hypersingular(
        dirichl_space, neumann_space, neumann_space, assembler=operator_assembler
    )
    adlp_in = laplace.adjoint_double_layer(
        neumann_space, neumann_space, neumann_space, assembler=operator_assembler
    )

    dlp_out = modified_helmholtz.double_layer(
        dirichl_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler
    )
    slp_out = modified_helmholtz.single_layer(
        neumann_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler
    )
    hlp_out = modified_helmholtz.hypersingular(
        dirichl_space, neumann_space, neumann_space, kappa, assembler=operator_assembler
    )
    adlp_out = modified_helmholtz.adjoint_double_layer(
        neumann_space, neumann_space, neumann_space, kappa, assembler=operator_assembler
    )

    phi_identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_identity = sparse.identity(neumann_space, neumann_space, neumann_space)

    ep = ep_ex / ep_in

    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-0.5 * (1 + alpha)) * phi_identity + (alpha * dlp_out) - dlp_in
    A[0, 1] = slp_in - ((alpha / ep) * slp_out)
    A[1, 0] = hlp_in - (beta * hlp_out)
    A[1, 1] = (
        (-0.5 * (1 + (beta / ep))) * dph_identity + adlp_in - ((beta / ep) * adlp_out)
    )

    self.matrices["A"] = A


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
            fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename=".rhs.tmp")
            tree = _laplace.setup(sources, targets, fmm)
            values = _laplace.evaluate(tree, fmm)
            os.remove(".rhs.tmp")
            result[:] = (-1.0) * values[:, 0] / ep_in

        @bempp.api.callable(vectorized=True)
        def fmm_d_green_func(x, n, domain_index, result):
            import exafmm.laplace as _laplace

            sources = _laplace.init_sources(x_q, q)
            targets = _laplace.init_targets(x.T)
            fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename=".rhs.tmp")
            tree = _laplace.setup(sources, targets, fmm)
            values = _laplace.evaluate(tree, fmm)
            os.remove(".rhs.tmp")
            result[:] = (-1.0) * np.sum(values[:, 1:] * n.T, axis=1) / ep_in

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=fmm_green_func)
        rhs_2 = bempp.api.GridFunction(dirichl_space, fun=fmm_d_green_func)

    else:

        @bempp.api.real_callable
        def d_green_func(x, n, domain_index, result):
            nrm = np.sqrt(
                (x[0] - x_q[:, 0]) ** 2
                + (x[1] - x_q[:, 1]) ** 2
                + (x[2] - x_q[:, 2]) ** 2
            )
            const = -1.0 / (4.0 * np.pi * ep_in)
            result[:] = (-1.0) * const * np.sum(q * np.dot(x - x_q, n) / (nrm ** 3))

        @bempp.api.real_callable
        def green_func(x, n, domain_index, result):
            nrm = np.sqrt(
                (x[0] - x_q[:, 0]) ** 2
                + (x[1] - x_q[:, 1]) ** 2
                + (x[2] - x_q[:, 2]) ** 2
            )
            result[:] = (-1.0) * np.sum(q / nrm) / (4.0 * np.pi * ep_in)

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=green_func)
        rhs_2 = bempp.api.GridFunction(dirichl_space, fun=d_green_func)

    self.rhs["rhs_1"], self.rhs["rhs_2"] = rhs_1, rhs_2


def laplace_multitrace(dirichl_space, neumann_space, operator_assembler):
    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-1.0) * laplace.double_layer(
        dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler
    )
    A[0, 1] = laplace.single_layer(
        neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler
    )
    A[1, 0] = laplace.hypersingular(
        dirichl_space, neumann_space, neumann_space, assembler=operator_assembler
    )
    A[1, 1] = laplace.adjoint_double_layer(
        neumann_space, neumann_space, neumann_space, assembler=operator_assembler
    )

    return A


def mod_helm_multitrace(dirichl_space, neumann_space, kappa, operator_assembler):
    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-1.0) * modified_helmholtz.double_layer(
        dirichl_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler
    )
    A[0, 1] = modified_helmholtz.single_layer(
        neumann_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler
    )
    A[1, 0] = modified_helmholtz.hypersingular(
        dirichl_space, neumann_space, neumann_space, kappa, assembler=operator_assembler
    )
    A[1, 1] = modified_helmholtz.adjoint_double_layer(
        neumann_space, neumann_space, neumann_space, kappa, assembler=operator_assembler
    )

    return A

def calculate_potential(self, rerun_all):
    calculate_potential_one_surface(self, rerun_all)
