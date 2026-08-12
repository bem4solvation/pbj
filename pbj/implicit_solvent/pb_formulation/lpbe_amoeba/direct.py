import numpy as np
import bempp_cl as bempp
import bempp_cl.api

"""Direct formulation specialized for AMOEBA polarizable electrostatics.

This module reuses the direct PB system but builds the right-hand-side terms for
AMOEBA multipoles and induced-dipole contributions.
"""

# import os
# from bempp_cl.api.operators.boundary import sparse, laplace, modified_helmholtz
from bempp_cl.api.operators.boundary import modified_helmholtz
import time
import pbj
import pbj.implicit_solvent.utils as utils

invert_potential = False


def verify_parameters(self):
    return True


def lhs(self):
    pbj.implicit_solvent.pb_formulation.lpbe.direct.lhs(self)


def rhs(self):

    force_field = self.force_field
    dirichl_space = self.dirichl_space
    neumann_space = self.neumann_space
    q = self.q
    x_q = self.x_q
    if force_field == "amoeba":
        d = self.d
        Q = self.Q
        # d_induced = self.d_induced
    ep_in = self.ep_in
    rhs_constructor = self.rhs_constructor

    if rhs_constructor == "fmm":

        @bempp.api.real_callable
        def multipolar_charges_fun(x, n, i, result):  # not using fmm
            T2 = np.zeros((len(x_q), 3, 3))
            phi = 0
            dist = x - x_q
            norm = np.sqrt(np.sum((dist * dist), axis=1))
            T0 = 1 / norm[:]
            T1 = np.transpose(dist.transpose() / norm**3)
            T2[:, :, :] = (
                np.ones((len(x_q), 3, 3))[:]
                * dist.reshape((len(x_q), 1, 3))
                * np.transpose(
                    np.ones((len(x_q), 3, 3)) * dist.reshape((len(x_q), 1, 3)),
                    (0, 2, 1),
                )
                / norm.reshape((len(x_q), 1, 1)) ** 5
            )
            phi = (
                np.sum(q[:] * T0[:])
                + np.sum(T1[:] * (d[:]))
                + 0.5 * np.sum(np.sum(T2[:] * Q[:], axis=1))
            )
            result[0] = phi / (4 * np.pi * ep_in)
            # only computes permanent multipole. Having a initial induced component is to be implemented

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=multipolar_charges_fun)

        coefs = np.zeros(neumann_space.global_dof_count)
        rhs_2 = bempp.api.GridFunction(neumann_space, coefficients=coefs)

    else:

        @bempp.api.real_callable
        def zero(x, n, domain_index, result):
            result[0] = 0

        @bempp.api.real_callable
        def multipolar_charges_fun(x, n, i, result):

            dist = np.zeros((3, len(x_q)))
            dist[0, :] = x[0] - x_q[:, 0]
            dist[1, :] = x[1] - x_q[:, 1]
            dist[2, :] = x[2] - x_q[:, 2]

            # dist = x - x_q
            # norm = np.sqrt(np.sum((dist*dist), axis = 1))

            norm = np.sqrt((dist[0, :]) ** 2 + (dist[1, :]) ** 2 + (dist[2, :]) ** 2)

            T0 = 1 / norm

            T1 = np.zeros((3, len(x_q)))
            T1[0, :] = dist[0, :]  # /norm**3
            T1[1, :] = dist[1, :]  # /norm**3
            T1[2, :] = dist[2, :]  # /norm**3

            T1 /= norm * norm * norm

            # T1 = np.transpose(dist.transpose()/norm**3)

            T2 = np.zeros((3, 3, len(x_q)))

            T2[0, 0, :] = dist[0, :] * dist[0, :] / norm**5
            T2[0, 1, :] = dist[0, :] * dist[1, :] / norm**5
            T2[0, 2, :] = dist[0, :] * dist[2, :] / norm**5
            T2[1, 1, :] = dist[1, :] * dist[1, :] / norm**5
            T2[1, 2, :] = dist[1, :] * dist[2, :] / norm**5
            T2[2, 2, :] = dist[2, :] * dist[2, :] / norm**5

            T2[2, 1, :] = T2[1, 2, :]
            T2[1, 0, :] = T2[0, 1, :]
            T2[2, 0, :] = T2[0, 2, :]

            # T2[:,:,:] = np.ones((len(x_q),3,3))[:]* dist.reshape((len(x_q),1,3))* \
            # np.transpose(np.ones((len(x_q),3,3))*dist.reshape((len(x_q),1,3)), (0,2,1))/norm.reshape((len(x_q),1,1))**5

            phi = (
                np.sum(q * T0)
                + np.sum(T1.transpose() * (d))
                + 0.5 * np.sum(np.sum(T2.transpose()[:] * Q[:], axis=1))
            )
            # only computes permanent multipole. Having a initial induced component is to be implemented
            result[0] = phi / (4 * np.pi * ep_in)

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=multipolar_charges_fun)
        rhs_2 = bempp.api.GridFunction(neumann_space, fun=zero)

    self.rhs["rhs_1"], self.rhs["rhs_2"] = rhs_1, rhs_2
    self.rhs["rhs_permanent_multipole_1"], self.rhs["rhs_permanent_multipole_2"] = (
        rhs_1,
        rhs_2,
    )


def rhs_induced_dipole(self):

    force_field = self.force_field
    dirichl_space = self.dirichl_space
    neumann_space = self.neumann_space
    x_q = self.x_q
    if force_field == "amoeba":
        d_induced = self.d_induced
    ep_in = self.ep_in
    rhs_constructor = self.rhs_constructor

    if rhs_constructor == "fmm":

        coefs = np.zeros(neumann_space.global_dof_count)

        @bempp.api.real_callable
        def dipole_charges_fun(x, n, i, result):  # not using fmm
            dist = x - x_q
            norm = np.sqrt(np.sum((dist * dist), axis=1))
            T1 = np.transpose(dist.transpose() / norm**3)
            phi = np.sum(T1[:] * d_induced[:])
            result[0] = phi / (4 * np.pi * ep_in)

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=dipole_charges_fun)

        # rhs_2 = bempp.api.GridFunction(neumann_space, fun=zero)
        rhs_2 = bempp.api.GridFunction(neumann_space, coefficients=coefs)

    else:

        @bempp.api.real_callable
        def zero(x, n, domain_index, result):
            result[0] = 0

        @bempp.api.real_callable
        def dipole_charges_fun(x, n, i, result):

            dist = np.zeros((3, len(x_q)))
            dist[0, :] = x[0] - x_q[:, 0]
            dist[1, :] = x[1] - x_q[:, 1]
            dist[2, :] = x[2] - x_q[:, 2]

            norm = np.sqrt((dist[0, :]) ** 2 + (dist[1, :]) ** 2 + (dist[2, :]) ** 2)

            T1 = np.zeros((3, len(x_q)))
            T1[0, :] = dist[0, :]
            T1[1, :] = dist[1, :]
            T1[2, :] = dist[2, :]

            T1 /= norm * norm * norm

            phi = np.sum(T1.transpose() * d_induced)
            result[0] = phi / (4 * np.pi * ep_in)

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=dipole_charges_fun)

        rhs_2 = bempp.api.GridFunction(neumann_space, fun=zero)

    # Add induced dipole component to already existing rhs with permanent multipoles
    self.rhs["rhs_1"] = self.rhs["rhs_permanent_multipole_1"] + rhs_1
    self.rhs["rhs_2"] = self.rhs["rhs_permanent_multipole_2"] + rhs_2


def block_diagonal_preconditioner(solute):
    pbj.implicit_solvent.pb_formulation.lpbe.direct.block_diagonal_preconditioner(
        solute
    )


def mass_matrix_preconditioner(solute):
    pbj.implicit_solvent.pb_formulation.lpbe.direct.mass_matrix_preconditioner(solute)


def mass_matrix_preconditioner_rhs(solute):
    pbj.implicit_solvent.pb_formulation.lpbe.direct.mass_matrix_preconditioner_rhs(
        solute
    )


def calculate_potential(simulation, rerun_all=False, rerun_rhs=False):

    start_time = time.time()

    for index, solute in enumerate(simulation.solutes):
        solute.results["induced_dipole"] = np.zeros_like(solute.d)

    if rerun_rhs and "A_discrete" in simulation.solutes[0].matrices:
        simulation.create_and_assemble_rhs()
    else:
        simulation.create_and_assemble_linear_system()

    simulation.timings["time_assembly"] = time.time() - start_time

    induced_dipole_residual = 1.0

    dipole_diff = np.zeros(len(simulation.solutes))

    dipole_iter_count = 0

    initial_guess = np.zeros_like(simulation.rhs["rhs_discrete"])

    simulation.timings["time_calc_gradient"] = 0.0
    simulation.timings["time_calc_induced_diss"] = 0.0
    simulation.timings["time_gmres"] = 0.0
    simulation.timings["time_assembly_rhs_induced_dipole"] = 0.0

    bempp.api.log(
        "PBJ: Starting self-consistent interations for induced dipole in dissolved state"
    )
    while induced_dipole_residual > simulation.induced_dipole_iter_tol:

        start_time_rhs = time.time()
        if dipole_iter_count != 0:
            create_and_assemble_rhs_induced_dipole(simulation)
        simulation.timings["time_assembly_rhs_induced_dipole"] += (
            time.time() - start_time_rhs
        )

        # Use GMRES to solve the system of equations
        if "preconditioning_matrix_gmres" in simulation.matrices:
            gmres_start_time = time.time()
            x, info, it_count = utils.solver(
                simulation.matrices["A_discrete"],
                simulation.rhs["rhs_discrete"],
                simulation.gmres_tolerance,
                simulation.gmres_restart,
                simulation.gmres_max_iterations,
                initial_guess=initial_guess,
                precond=simulation.matrices["preconditioning_matrix_gmres"],
            )

        else:
            gmres_start_time = time.time()
            x, info, it_count = utils.solver(
                simulation.matrices["A_discrete"],
                simulation.rhs["rhs_discrete"],
                simulation.gmres_tolerance,
                simulation.gmres_restart,
                simulation.gmres_max_iterations,
                initial_guess=initial_guess,
            )

        simulation.timings["time_gmres"] += time.time() - gmres_start_time

        initial_guess = x.copy()

        from bempp_cl.api.assembly.blocked_operator import (
            grid_function_list_from_coefficients,
        )

        solute_start = 0
        for index, solute in enumerate(simulation.solutes):

            N_dirichl = solute.dirichl_space.global_dof_count
            N_neumann = solute.neumann_space.global_dof_count
            N_total = N_dirichl + N_neumann

            x_slice = x.ravel()[solute_start : solute_start + N_total]

            solute_start += N_total

            solution = grid_function_list_from_coefficients(
                x_slice, simulation.solutes[index].matrices["A"].domain_spaces
            )

            solute.results["phi"] = solution[0]

            if simulation.formulation_object.invert_potential:
                solute.results["d_phi"] = (solute.ep_ex / solute.ep_in) * solution[1]
            else:
                solute.results["d_phi"] = solution[1]

            time_start_grad = time.time()
            solute.calculate_gradient_field()
            simulation.timings["time_calc_gradient"] += time.time() - time_start_grad

            d_induced_prev = solute.results["induced_dipole"].copy()

            time_start_induced = time.time()
            solute.calculate_induced_dipole_dissolved()
            simulation.timings["time_calc_induced_diss"] += (
                time.time() - time_start_induced
            )

            d_induced = solute.results["induced_dipole"]

            dipole_diff[index] = np.max(
                np.sqrt(
                    np.sum((np.linalg.norm(d_induced_prev - d_induced, axis=1)) ** 2)
                    / len(d_induced)
                )
            )

        induced_dipole_residual = np.max(dipole_diff)

        bempp.api.log(
            "PBJ: Dissolved induced dipole iteration %i -> residual: %s"
            % (dipole_iter_count, induced_dipole_residual)
        )

        dipole_iter_count += 1

    simulation.timings["time_compute_potential"] = time.time() - start_time


def create_and_assemble_rhs_induced_dipole(simulation):

    rhs_final_discrete = []

    for index, solute in enumerate(simulation.solutes):

        initialise_rhs_induced_dipole(simulation, solute)
        solute.apply_preconditioning_rhs()

        simulation.rhs["rhs_" + str(index + 1)] = [
            solute.rhs["rhs_1"],
            solute.rhs["rhs_2"],
        ]

        rhs_final_discrete.extend(solute.rhs["rhs_discrete"])

    simulation.rhs["rhs_discrete"] = rhs_final_discrete


def lhs_inter_solute_interactions(simulation, solute_target, solute_source):

    dirichl_space_target = solute_target.dirichl_space
    neumann_space_target = solute_target.neumann_space
    dirichl_space_source = solute_source.dirichl_space
    neumann_space_source = solute_source.neumann_space

    ep_in = solute_source.ep_in
    ep_out = simulation.ep_ex
    kappa = simulation.kappa
    operator_assembler = simulation.operator_assembler

    dlp = modified_helmholtz.double_layer(
        dirichl_space_source,
        dirichl_space_target,
        dirichl_space_target,
        kappa,
        assembler=operator_assembler,
    )
    slp = modified_helmholtz.single_layer(
        neumann_space_source,
        neumann_space_target,
        neumann_space_target,
        kappa,
        assembler=operator_assembler,
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
    A_inter[1, 0] = -dlp
    A_inter[1, 1] = (ep_in / ep_out) * slp

    solute_target.matrices["A_inter"].append(A_inter)

    # return A_inter.weak_form()  # should always be weak_form, as preconditioner doesn't touch it


def initialise_rhs_induced_dipole(simulation, solute):
    start_rhs = time.time()
    # Verify if parameters are already set and then save RHS
    if solute.formulation_object.verify_parameters(solute):
        solute.formulation_object.rhs_induced_dipole(solute)
    simulation.timings["time_rhs_initialisation"] = time.time() - start_rhs
