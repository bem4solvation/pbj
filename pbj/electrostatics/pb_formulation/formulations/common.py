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
        if "preconditioning_matrix_gmres" in self.matrices:
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


def calculate_solvation_energy_one_surface(self, rerun_all):
        if rerun_all:
            self.calculate_potential(rerun_all)

        if "phi" not in self.results:
            # If surface potential has not been calculated, calculate it now
            self.calculate_potential()

        start_time = time.time()

        solution_dirichl = self.results["phi"]
        solution_neumann = self.results["d_phi"]

        from bempp.api.operators.potential.laplace import single_layer, double_layer

        slp_q = single_layer(self.neumann_space, self.x_q.transpose())
        dlp_q = double_layer(self.dirichl_space, self.x_q.transpose())
        phi_q = slp_q * solution_neumann - dlp_q * solution_dirichl

        # total solvation energy applying constant to get units [kcal/mol]
        total_energy = 2 * np.pi * 332.064 * np.sum(self.q * phi_q).real
        self.results["solvation_energy"] = total_energy
        self.timings["time_calc_energy"] = time.time() - start_time

        if self.print_times:
            print(
                "It took ",
                self.timings["time_calc_energy"],
                " seconds to compute the solvation energy",
            )


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
        if "preconditioning_matrix_gmres" in self.matrices:
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

        (dirichlet_diel_solution, neumann_diel_solution, dirichlet_stern_solution, neumann_stern_solution) = grid_function_list_from_coefficients(
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
