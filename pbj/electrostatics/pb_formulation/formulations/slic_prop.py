import numpy as np
import bempp.api
from .slic import calculate_potential_slic ############ maybe move to .common?????? CHECK
import pbj


def verify_parameters(self):
    return True


def lhs(self):
    pbj.electrostatics.pb_formulation.formulations.direct_stern.lhs(self)


def rhs(self):
    pbj.electrostatics.pb_formulation.formulations.direct_stern.rhs(self)


def block_diagonal_preconditioner(self):
    pbj.electrostatics.pb_formulation.formulations.direct_stern.block_diagonal_preconditioner(
        self
    )


def mass_matrix_preconditioner(self):
    pbj.electrostatics.pb_formulation.formulations.direct_stern.mass_matrix_preconditioner(
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


def calculate_potential(self, rerun_all):
    dirichl_space_diel = self.dirichl_space

    ep_stern = getattr(self, "ep_stern", self.ep_ex)
    self.ep_stern = ep_stern

    if self.stern_object is None:
        pbj.electrostatics.pb_formulation.formulations.direct_stern.create_stern_mesh(
            self
        )

    max_iterations = self.slic_max_iterations
    tolerance = self.slic_tolerance

    it = 0
    phi_L2error = 1.0

    sigma = bempp.api.GridFunction(
        dirichl_space_diel, coefficients=np.zeros(dirichl_space_diel.global_dof_count)
    )

    self.timings["time_gmres"] = []

    self.timings["time_compute_potential"] = []

    self.results["solver_iteration_count"] = []

    time_matrix_initialisation = []
    time_matrix_assembly = []
    time_preconditioning = []

    self.initialise_rhs()

    while it < max_iterations and phi_L2error > tolerance:

        if it == 0:
            self.e_hat_diel = self.ep_in / self.ep_stern
            self.e_hat_stern = self.ep_stern / self.ep_ex

        else:
            pbj.electrostatics.pb_formulation.formulations.slic.create_ehat_diel(
                self, sigma
            )
            create_ehat_stern(self)
            phi_old = self.results["phi"].coefficients.copy()

        calculate_potential_slic(self)

        sigma = pbj.electrostatics.pb_formulation.formulations.slic.solve_sigma(self)

        if it != 0:
            phi_L2error = np.sqrt(
                np.sum((phi_old - self.results["phi"].coefficients) ** 2)
                / np.sum(self.results["phi"].coefficients ** 2)
            )

        it += 1

        time_matrix_initialisation.append(self.timings["time_matrix_initialisation"])
        time_matrix_assembly.append(self.timings["time_matrix_assembly"])
        time_preconditioning.append(self.timings["time_preconditioning"])

    self.timings["time_matrix_initialisation"] = time_matrix_initialisation
    self.timings["time_matrix_assembly"] = time_matrix_assembly
    self.timings["time_preconditioning"] = time_preconditioning
