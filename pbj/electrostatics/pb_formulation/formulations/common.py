import bempp.api
import numpy as np
import time
import pbj.electrostatics.utils as utils


def calculate_potential_one_surface(self, rerun_all):
    # Start the overall timing for the whole process
    start_time = time.time()

    if rerun_all:
        self.initialise_matrices()
        self.assemble_matrices()
        self.initialise_rhs()
        self.apply_preconditioning()
        # self.pass_to_discrete_form()

    else:
        if "A" not in self.matrices or "rhs_1" not in self.rhs:
            # If matrix A or rhs_1 doesn't exist, it must first be created
            self.initialise_matrices()
            self.initialise_rhs()
        if not self.matrices["A"]._cached:
            self.assemble_matrices()
        if "A_discrete" not in self.matrices or "rhs_discrete" not in self.rhs:
            # See if preconditioning needs to be applied if this hasn't been done
            self.apply_preconditioning()
        # if "A_discrete" not in self.matrices or "rhs_discrete" not in self.rhs:
        #   # See if discrete form has been called
        #  self.pass_to_discrete_form()

    # Use GMRES to solve the system of equations
    gmres_start_time = time.time()
    if (
        "preconditioning_matrix_gmres" in self.matrices
        and self.pb_formulation_preconditioning == True
    ):
        x, info, it_count = utils.solver(
            self.matrices["A_discrete"],
            self.rhs["rhs_discrete"],
            self.gmres_tolerance,
            self.gmres_restart,
            self.gmres_max_iterations,
            precond=self.matrices["preconditioning_matrix_gmres"],
        )
    else:
        x, info, it_count = utils.solver(
            self.matrices["A_discrete"],
            self.rhs["rhs_discrete"],
            self.gmres_tolerance,
            self.gmres_restart,
            self.gmres_max_iterations,
        )

    self.timings["time_gmres"] = time.time() - gmres_start_time

    # Split solution and generate corresponding grid functions
    from bempp.api.assembly.blocked_operator import (
        grid_function_list_from_coefficients,
    )

    (dirichlet_solution, neumann_solution) = grid_function_list_from_coefficients(
        x.ravel(), self.matrices["A"].domain_spaces
    )

    # Save number of iterations taken and the solution of the system
    self.results["solver_iteration_count"] = it_count
    self.results["phi"] = dirichlet_solution
    if self.formulation_object.invert_potential:
        self.results["d_phi"] = (self.ep_ex / self.ep_in) * neumann_solution
    else:
        self.results["d_phi"] = neumann_solution

    # Finished computing surface potential, register total time taken
    self.timings["time_compute_potential"] = time.time() - start_time

    # Print times, if this is desired
    if self.print_times:
        show_potential_calculation_times(self)


def calculate_potential_stern(self, rerun_all):

    # Start the overall timing for the whole process
    start_time = time.time()

    if rerun_all:
        self.initialise_matrices()
        self.assemble_matrices()
        self.initialise_rhs()
        self.apply_preconditioning()
        # self.pass_to_discrete_form()

    else:
        if "A" not in self.matrices or "rhs_1" not in self.rhs:
            # If matrix A or rhs_1 doesn't exist, it must first be created
            self.initialise_matrices()
            self.initialise_rhs()
        if not self.matrices["A"]._cached:
            self.assemble_matrices()
        if "A_discrete" not in self.matrices or "rhs_discrete" not in self.rhs:
            # See if preconditioning needs to be applied if this hasn't been done
            self.apply_preconditioning()
        # if "A_discrete" not in self.matrices or "rhs_discrete" not in self.rhs:
        #   # See if discrete form has been called
        #  self.pass_to_discrete_form()

    # Use GMRES to solve the system of equations
    gmres_start_time = time.time()
    if (
        "preconditioning_matrix_gmres" in self.matrices
        and self.pb_formulation_preconditioning == True
    ):
        x, info, it_count = utils.solver(
            self.matrices["A_discrete"],
            self.rhs["rhs_discrete"],
            self.gmres_tolerance,
            self.gmres_restart,
            self.gmres_max_iterations,
            precond=self.matrices["preconditioning_matrix_gmres"],
        )
    else:
        x, info, it_count = utils.solver(
            self.matrices["A_discrete"],
            self.rhs["rhs_discrete"],
            self.gmres_tolerance,
            self.gmres_restart,
            self.gmres_max_iterations,
        )

    self.timings["time_gmres"] = time.time() - gmres_start_time

    # Split solution and generate corresponding grid functions
    from bempp.api.assembly.blocked_operator import (
        grid_function_list_from_coefficients,
    )

    (
        dirichlet_diel_solution,
        neumann_diel_solution,
        dirichlet_stern_solution,
        neumann_stern_solution,
    ) = grid_function_list_from_coefficients(
        x.ravel(), self.matrices["A"].domain_spaces
    )

    # Save number of iterations taken and the solution of the system
    self.results["solver_iteration_count"] = it_count
    self.results["phi"] = dirichlet_diel_solution
    self.results["d_phi"] = neumann_diel_solution
    self.results["phi_stern"] = dirichlet_stern_solution
    self.results["d_phi_stern"] = neumann_stern_solution

    # Finished computing surface potential, register total time taken
    self.timings["time_compute_potential"] = time.time() - start_time

    # Print times, if this is desired
    if self.print_times:
        show_potential_calculation_times(self)


def calculate_potential_slic(self):

    # Start the overall timing for one SLIC iteration
    start_time = time.time()

    self.initialise_matrices()
    self.assemble_matrices()
    self.apply_preconditioning()

    # Use GMRES to solve the system of equations
    gmres_start_time = time.time()
    if (
        "preconditioning_matrix_gmres" in self.matrices
        and self.pb_formulation_preconditioning == True
    ):
        x, info, it_count = utils.solver(
            self.matrices["A_discrete"],
            self.rhs["rhs_discrete"],
            self.gmres_tolerance,
            self.gmres_restart,
            self.gmres_max_iterations,
            precond=self.matrices["preconditioning_matrix_gmres"],
        )
    else:
        x, info, it_count = utils.solver(
            self.matrices["A_discrete"],
            self.rhs["rhs_discrete"],
            self.gmres_tolerance,
            self.gmres_restart,
            self.gmres_max_iterations,
        )

    self.timings["time_gmres"].append(time.time() - gmres_start_time)

    # Split solution and generate corresponding grid functions
    from bempp.api.assembly.blocked_operator import (
        grid_function_list_from_coefficients,
    )

    (
        dirichlet_diel_solution,
        neumann_diel_solution,
        dirichlet_stern_solution,
        neumann_stern_solution,
    ) = grid_function_list_from_coefficients(
        x.ravel(), self.matrices["A"].domain_spaces
    )

    # Save number of iterations taken and the solution of the system
    self.results["solver_iteration_count"].append(it_count)
    self.results["phi"] = dirichlet_diel_solution
    self.results["d_phi"] = neumann_diel_solution
    self.results["phi_stern"] = dirichlet_stern_solution
    self.results["d_phi_stern"] = neumann_stern_solution

    # Finished computing surface potential, register total time taken
    self.timings["time_compute_potential"].append(time.time() - start_time)

    # Print times, if this is desired
    if self.print_times:
        show_potential_calculation_times(self)


def show_potential_calculation_times(self):
    if "phi" in self.results:
        print(
            "It took ",
            self.timings["time_matrix_construction"],
            " seconds to construct the matrices",
        )
        print(
            "It took ",
            self.timings["time_rhs_construction"],
            " seconds to construct the rhs vectors",
        )
        print(
            "It took ",
            self.timings["time_matrix_to_discrete"],
            " seconds to pass the main matrix to discrete form ("
            + self.discrete_form_type
            + ")",
        )
        print(
            "It took ",
            self.timings["time_preconditioning"],
            " seconds to compute and apply the preconditioning ("
            + str(self.pb_formulation_preconditioning)
            + "("
            + self.pb_formulation_preconditioning_type
            + ")",
        )
        print(
            "It took ",
            self.timings["time_gmres"],
            " seconds to resolve the system using GMRES",
        )
        print(
            "It took ",
            self.timings["time_compute_potential"],
            " seconds in total to compute the surface potential",
        )
    else:
        print("Potential must first be calculated to show times.")
