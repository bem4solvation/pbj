import numpy as np
import bempp.api
import os
from bempp.api.operators.boundary import laplace, modified_helmholtz
from .common import calculate_potential_one_surface

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

    dlp_ex = modified_helmholtz.double_layer(
        dirichl_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler
    )
    slp_ex = modified_helmholtz.single_layer(
        neumann_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler
    )
    hlp_ex = modified_helmholtz.hypersingular(
        dirichl_space, neumann_space, neumann_space, kappa, assembler=operator_assembler
    )
    adlp_ex = modified_helmholtz.adjoint_double_layer(
        neumann_space, neumann_space, neumann_space, kappa, assembler=operator_assembler
    )

    ep = ep_ex / ep_in

    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-1.0 * dlp_ex) - dlp_in
    A[0, 1] = slp_ex + (ep * slp_in)
    A[1, 0] = hlp_ex + ((1.0 / ep) * hlp_in)
    A[1, 1] = adlp_ex + adlp_in

    calderon_int_scal = bempp.api.BlockedOperator(2, 2)
    calderon_int_scal[0, 0] = -1.0 * dlp_in
    calderon_int_scal[0, 1] = ep * slp_in
    calderon_int_scal[1, 0] = (1.0 / ep) * hlp_in
    calderon_int_scal[1, 1] = adlp_in

    calderon_ext = bempp.api.BlockedOperator(2, 2)
    calderon_ext[0, 0] = -1.0 * dlp_ex
    calderon_ext[0, 1] = slp_ex
    calderon_ext[1, 0] = hlp_ex
    calderon_ext[1, 1] = adlp_ex

    self.matrices["A"], self.matrices["A_int_scal"], self.matrices["A_ext"] = (
        A,
        calderon_int_scal,
        calderon_ext,
    )


def calderon_A_only(solute):
    dirichl_space = solute.dirichl_space
    neumann_space = solute.neumann_space
    ep_in = solute.ep_in
    ep_ex = solute.ep_ex
    kappa = solute.kappa
    operator_assembler = solute.operator_assembler

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

    dlp_ex = modified_helmholtz.double_layer(
        dirichl_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler
    )
    slp_ex = modified_helmholtz.single_layer(
        neumann_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler
    )
    hlp_ex = modified_helmholtz.hypersingular(
        dirichl_space, neumann_space, neumann_space, kappa, assembler=operator_assembler
    )
    adlp_ex = modified_helmholtz.adjoint_double_layer(
        neumann_space, neumann_space, neumann_space, kappa, assembler=operator_assembler
    )

    ep = ep_ex / ep_in

    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = (-1.0 * dlp_ex) - dlp_in
    A[0, 1] = slp_ex + (ep * slp_in)
    A[1, 0] = hlp_ex + ((1.0 / ep) * hlp_in)
    A[1, 1] = adlp_ex + adlp_in

    return A


def rhs(self):
    dirichl_space = self.dirichl_space
    neumann_space = self.neumann_space
    q = self.q
    x_q = self.x_q
    ep_in = self.ep_in
    ep_ex = self.ep_ex
    rhs_constructor = self.rhs_constructor

    if rhs_constructor == "fmm":

        @bempp.api.callable(vectorized=True)
        def rhs1_fun(x, n, domain_index, result):
            import exafmm.laplace as _laplace

            sources = _laplace.init_sources(x_q, q)
            targets = _laplace.init_targets(x.T)
            fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename=".rhs.tmp")
            tree = _laplace.setup(sources, targets, fmm)
            values = _laplace.evaluate(tree, fmm)
            os.remove(".rhs.tmp")
            result[:] = (-1.0) * values[:, 0] / ep_in

        @bempp.api.callable(vectorized=True)
        def rhs2_fun(x, n, domain_index, result):
            import exafmm.laplace as _laplace

            sources = _laplace.init_sources(x_q, q)
            targets = _laplace.init_targets(x.T)
            fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename=".rhs.tmp")
            tree = _laplace.setup(sources, targets, fmm)
            values = _laplace.evaluate(tree, fmm)
            os.remove(".rhs.tmp")
            result[:] = (-1.0) * np.sum(values[:, 1:] * n.T, axis=1) / ep_ex

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=rhs1_fun)
        rhs_2 = bempp.api.GridFunction(neumann_space, fun=rhs2_fun)

    else:

        @bempp.api.real_callable
        def d_green_func(x, n, domain_index, result):
            nrm = np.sqrt(
                (x[0] - x_q[:, 0]) ** 2
                + (x[1] - x_q[:, 1]) ** 2
                + (x[2] - x_q[:, 2]) ** 2
            )
            const = -1.0 / (4.0 * np.pi * ep_in)
            result[:] = (
                -1.0
                * (ep_in / ep_ex)
                * const
                * np.sum(q * np.dot(x - x_q, n) / (nrm**3))
            )

        @bempp.api.real_callable
        def green_func(x, n, domain_index, result):
            nrm = np.sqrt(
                (x[0] - x_q[:, 0]) ** 2
                + (x[1] - x_q[:, 1]) ** 2
                + (x[2] - x_q[:, 2]) ** 2
            )
            result[:] = -1.0 * np.sum(q / nrm) / (4.0 * np.pi * ep_in)

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=green_func)
        rhs_2 = bempp.api.GridFunction(neumann_space, fun=d_green_func)

    self.rhs["rhs_1"], self.rhs["rhs_2"] = rhs_1, rhs_2


def mass_matrix_preconditioner(solute):
    from pbj.electrostatics.solute import matrix_to_discrete_form, rhs_to_discrete_form

    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]
    solute.matrices["A_discrete"] = matrix_to_discrete_form(
        solute.matrices["A_final"], "strong"
    )
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(
        solute.rhs["rhs_final"], "strong", solute.matrices["A"]
    )


def calderon_squared_preconditioner(solute):
    solute.matrices["preconditioning_matrix"] = solute.matrices["A"]
    apply_calderon_precondtioning(solute)


def calderon_interior_operator_scaled_preconditioner(solute):
    solute.matrices["preconditioning_matrix"] = solute.matrices["A_int_scal"]
    apply_calderon_precondtioning(solute)


def calderon_exterior_operator_preconditioner(solute):
    solute.matrices["preconditioning_matrix"] = solute.matrices["A_ext"]
    apply_calderon_precondtioning(solute)


def apply_calderon_precondtioning(solute):
    from pbj.electrostatics.solute import matrix_to_discrete_form, rhs_to_discrete_form

    solute.matrices["A_final"] = (
        solute.matrices["preconditioning_matrix"] * solute.matrices["A"]
    )
    solute.rhs["rhs_final"] = solute.matrices["preconditioning_matrix"] * [
        solute.rhs["rhs_1"],
        solute.rhs["rhs_2"],
    ]

    solute.matrices["A_discrete"] = matrix_to_discrete_form(
        solute.matrices["A_final"], "strong"
    )
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(
        solute.rhs["rhs_final"], "strong", solute.matrices["A"]
    )


def calderon_interior_operator_scaled_with_scaled_mass_matrix_preconditioner(solute):
    from bempp.api.utils.helpers import get_inverse_mass_matrix
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
    from pbj.electrostatics.solute import matrix_to_discrete_form, rhs_to_discrete_form

    preconditioner = solute.matrices["A_int_scal"]
    ep_ex = solute.ep_ex
    ep_in = solute.ep_in

    nrows = len(preconditioner.range_spaces)
    range_ops = np.empty((nrows, nrows), dtype="O")

    for index in range(nrows):
        range_ops[index, index] = get_inverse_mass_matrix(
            preconditioner.range_spaces[index],
            preconditioner.dual_to_range_spaces[index],
        )

    range_ops[0, 0] = range_ops[0, 0] * (1.0 / (0.25 + (ep_ex / (4.0 * ep_in))))
    range_ops[1, 1] = range_ops[1, 1] * (1.0 / (0.25 + (ep_in / (4.0 * ep_ex))))

    mass_matrix = BlockedDiscreteOperator(range_ops)

    preconditioner_with_mass = mass_matrix * preconditioner.weak_form()

    solute.matrices["preconditioning_matrix"] = preconditioner_with_mass
    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]

    solute.matrices["A_discrete"] = matrix_to_discrete_form(
        solute.matrices["A_final"], "strong"
    )
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(
        solute.rhs["rhs_final"], "strong", solute.matrices["A"]
    )

    solute.matrices["A_discrete"] = (
        solute.matrices["preconditioning_matrix"] * solute.matrices["A_discrete"]
    )
    solute.rhs["rhs_discrete"] = (
        solute.matrices["preconditioning_matrix"] * solute.rhs["rhs_discrete"]
    )


def calderon_exterior_operator_with_scaled_mass_matrix_preconditioner(solute):
    from bempp.api.utils.helpers import get_inverse_mass_matrix
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator
    from pbj.electrostatics.solute import matrix_to_discrete_form, rhs_to_discrete_form

    preconditioner = solute.matrices["A_ext"]
    ep_ex = solute.ep_ex
    ep_in = solute.ep_in

    nrows = len(preconditioner.range_spaces)
    range_ops = np.empty((nrows, nrows), dtype="O")

    for index in range(nrows):
        range_ops[index, index] = get_inverse_mass_matrix(
            preconditioner.range_spaces[index],
            preconditioner.dual_to_range_spaces[index],
        )

    range_ops[0, 0] = range_ops[0, 0] * (1.0 / (0.25 + (ep_in / (4.0 * ep_ex))))
    range_ops[1, 1] = range_ops[1, 1] * (1.0 / (0.25 + (ep_ex / (4.0 * ep_in))))

    mass_matrix = BlockedDiscreteOperator(range_ops)

    preconditioner_with_mass = mass_matrix * preconditioner.weak_form()

    solute.matrices["preconditioning_matrix"] = preconditioner_with_mass
    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]

    solute.matrices["A_discrete"] = matrix_to_discrete_form(
        solute.matrices["A_final"], "strong"
    )
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(
        solute.rhs["rhs_final"], "strong", solute.matrices["A"]
    )

    solute.matrices["A_discrete"] = (
        solute.matrices["preconditioning_matrix"] * solute.matrices["A_discrete"]
    )
    solute.rhs["rhs_discrete"] = (
        solute.matrices["preconditioning_matrix"] * solute.rhs["rhs_discrete"]
    )


def calderon_squared_lowered_parameters_preconditioner(solute):
    from pbj.electrostatics.solute import rhs_to_discrete_form

    # Pass A to strong form with user set parameters
    solute.matrices["A"].strong_form()

    # Save current set parameters to be reset later
    expansion_order_main = bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order
    ncrit_main = bempp.api.GLOBAL_PARAMETERS.fmm.ncrit
    reg_quadrature_points_main = bempp.api.GLOBAL_PARAMETERS.quadrature.regular
    sing_quadrature_points_main = bempp.api.GLOBAL_PARAMETERS.quadrature.singular

    # Change parameters
    bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order = 2
    bempp.api.GLOBAL_PARAMETERS.fmm.ncrit = 50
    bempp.api.GLOBAL_PARAMETERS.quadrature.regular = 1
    bempp.api.GLOBAL_PARAMETERS.quadrature.singular = 3

    solute.matrices["preconditioning_matrix"] = calderon_A_only(solute)

    solute.matrices["preconditioning_matrix"].strong_form()
    solute.matrices["A_discrete"] = (
        solute.matrices["preconditioning_matrix"].strong_form()
        * solute.matrices["A"].strong_form()
    )

    bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order = expansion_order_main
    bempp.api.GLOBAL_PARAMETERS.fmm.ncrit = ncrit_main
    bempp.api.GLOBAL_PARAMETERS.quadrature.regular = reg_quadrature_points_main
    bempp.api.GLOBAL_PARAMETERS.quadrature.singular = sing_quadrature_points_main

    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]
    solute.rhs["rhs_discrete"] = solute.matrices[
        "preconditioning_matrix"
    ].strong_form() * rhs_to_discrete_form(
        solute.rhs["rhs_final"], "strong", solute.matrices["A"]
    )


def calculate_potential(self, rerun_all):
    calculate_potential_one_surface(self, rerun_all)
