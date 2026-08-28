import numpy as np
import bempp_cl as bempp
import bempp_cl.api

"""SLIC-prop formulation for the Stern-layer PB problem.

This module extends SLIC by updating additional Stern-layer properties during the
iterative solution process.
"""

from .direct import (
    calculate_potential_slic,
)  # maybe move to .common?????? CHECK
import pbj
from ..lpbe.common import calculate_potential_stern

invert_potential = False


def verify_parameters(self):
    return True


def lhs(self):
    pbj.implicit_solvent.pb_formulation.lpbe.direct_stern.lhs(self)


def rhs(self):
    pbj.implicit_solvent.pb_formulation.lpbe.direct_stern.rhs(self)


def block_diagonal_preconditioner(self):
    pbj.implicit_solvent.pb_formulation.lpbe.direct_stern.block_diagonal_preconditioner(
        self
    )


def mass_matrix_preconditioner(self):
    pbj.implicit_solvent.pb_formulation.lpbe.direct_stern.mass_matrix_preconditioner(
        self
    )


def create_ehat_stern(self):
    neumann_space_stern = self.stern_object.neumann_space

    x_q = self.x_q
    q = self.q
    ep_stern = getattr(self, "ep_stern", self.ep_ex)
    self.ep_stern = ep_stern

    @bempp.api.real_callable
    def d1_function(x, n, domain_index, result):
        nrm = np.sqrt(
            (x[0] - x_q[:, 0]) ** 2 + (x[1] - x_q[:, 1]) ** 2 + (x[2] - x_q[:, 2]) ** 2
        )
        result[:] = np.sum(q / nrm)

    d1_fun = bempp.api.GridFunction(neumann_space_stern, fun=d1_function)
    if np.sum(q) < 1e-8:
        d1_mat = -(1 / ep_stern) * d1_fun.coefficients
    else:
        d1_mat = (
            -(np.sum(q) / ep_stern) * d1_fun.coefficients / np.mean(d1_fun.coefficients)
        )
    d1_gridfun = bempp.api.GridFunction(neumann_space_stern, coefficients=d1_mat)
    d1_op = bempp.api.assembly.boundary_operator.MultiplicationOperator(
        d1_gridfun, neumann_space_stern, neumann_space_stern, neumann_space_stern
    )
    d2 = self.results["d_phi_stern"].integrate()[0]
    self.e_hat_stern = (1 / d2) * d1_op


def calculate_potential(simulation, rerun_all, rerun_rhs):

    if len(simulation.solutes) > 1:
        print("Direct prop only available for one solute")
        return

    solute = simulation.solutes[0]
    dirichl_space_diel = solute.dirichl_space

    ep_stern = getattr(solute, "ep_stern", solute.ep_ex)
    solute.ep_stern = ep_stern

    if solute.stern_object is None:
        pbj.implicit_solvent.pb_formulation.lpbe.direct_stern.create_stern_mesh(solute)

    max_iterations = solute.slic_max_iterations
    tolerance = solute.slic_tolerance

    it = 0
    phi_L2error = 1.0

    sigma = bempp.api.GridFunction(
        dirichl_space_diel, coefficients=np.zeros(dirichl_space_diel.global_dof_count)
    )
    # Store initial sigma on the solute so solve_sigma can use it as x0
    solute.slic_sigma = sigma

    solute.timings["time_gmres"] = []
    solute.timings["time_compute_potential"] = []
    solute.results["solver_iteration_count"] = []

    time_matrix_initialisation = []
    time_matrix_assembly = []
    time_preconditioning = []

    # Assemble initial system (matrices + discrete RHS) so rhs_discrete exists
    calculate_potential_stern(simulation)

    while it < max_iterations and phi_L2error > tolerance:

        if it == 0:
            solute.e_hat_diel = solute.ep_in / solute.ep_stern
            solute.e_hat_stern = solute.ep_stern / solute.ep_ex

        else:
            pbj.implicit_solvent.pb_formulation.lpbe_slic.direct.create_ehat_diel(
                solute
            )
            create_ehat_stern(solute)
            phi_old = solute.results["phi"].coefficients.copy()

        # reuse the SLIC solver which expects the full simulation object
        calculate_potential_slic(simulation)

        sigma = pbj.implicit_solvent.pb_formulation.lpbe_slic.direct.solve_sigma(solute)

        if it != 0:
            phi_L2error = np.sqrt(
                np.sum((phi_old - solute.results["phi"].coefficients) ** 2)
                / np.sum(solute.results["phi"].coefficients ** 2)
            )

        it += 1

        time_matrix_initialisation.append(solute.timings["time_matrix_initialisation"])
        time_matrix_assembly.append(simulation.timings.get("time_assembly", 0))
        time_preconditioning.append(solute.timings["time_preconditioning"])

    solute.timings["time_matrix_initialisation"] = time_matrix_initialisation
    solute.timings["time_matrix_assembly"] = time_matrix_assembly
    solute.timings["time_preconditioning"] = time_preconditioning
