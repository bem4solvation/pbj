# import bempp.api
import numpy as np
import time
import pbj.electrostatics.utils as utils
        
def calculate_potential_one_surface(simulation, rerun_all=False, rerun_rhs=False):

        if rerun_all and rerun_rhs: # if both are True, just rerun_all
            rerun_rhs=False
            
        #if self.solutes[0].force_field == "amoeba":
        #    self.calculate_potentials_polarizable(rerun_all=rerun_all, rerun_rhs=rerun_rhs)

        elif ("phi" not in simulation.solutes[0].results) or (rerun_all) or (rerun_rhs):
            start_time = time.time()

            if rerun_rhs and "A_discrete" in simulation.solutes[0].matrices:
                simulation.create_and_assemble_rhs()
            else:
                simulation.create_and_assemble_linear_system()
            
            simulation.timings["time_assembly"] = time.time() - start_time 
           
            initial_guess = np.zeros_like(simulation.rhs["rhs_discrete"])

            # Use GMRES to solve the system of equations
            if "preconditioning_matrix_gmres" in simulation.matrices:
                gmres_start_time = time.time()
                x, info, it_count = utils.solver(
                    simulation.matrices["A_discrete"],
                    simulation.rhs["rhs_discrete"],
                    simulation.gmres_tolerance,
                    simulation.gmres_restart,
                    simulation.gmres_max_iterations,
                    initial_guess = initial_guess,
                    precond = simulation.matrices["preconditioning_matrix_gmres"]
                )
                
            else:
                gmres_start_time = time.time()
                x, info, it_count = utils.solver(
                    simulation.matrices["A_discrete"],
                    simulation.rhs["rhs_discrete"],
                    simulation.gmres_tolerance,
                    simulation.gmres_restart,
                    simulation.gmres_max_iterations,
                    initial_guess = initial_guess,
                )

            simulation.timings["time_gmres"] = time.time() - gmres_start_time

            from bempp.api.assembly.blocked_operator import (
                grid_function_list_from_coefficients,
            )          

            simulation.run_info["solver_iteration_count"] = it_count
            
            solute_start = 0
            for index, solute in enumerate(simulation.solutes):

                N_dirichl = solute.dirichl_space.global_dof_count
                N_neumann = solute.neumann_space.global_dof_count
                N_total = N_dirichl + N_neumann
                
                x_slice = x.ravel()[solute_start:solute_start + N_total]
                
                solute_start += N_total
                
                solution = grid_function_list_from_coefficients(
                    x_slice, simulation.solutes[index].matrices["A"].domain_spaces
                )

                solute.results["phi"]  = solution[0]
                
                if simulation.formulation_object.invert_potential:
                    solute.results["d_phi"] = (solute.ep_ex / solute.ep_in) * solution[1] 
                else:  
                    solute.results["d_phi"] = solution[1] 
  
            simulation.timings["time_compute_potential"] = time.time() - start_time

    
def calculate_potential_stern(simulation, rerun_all=False, rerun_rhs=False):

        if rerun_all and rerun_rhs: # if both are True, just rerun_all
            rerun_rhs=False
         
        elif ("phi" not in simulation.solutes[0].results) or (rerun_all) or (rerun_rhs):
            start_time = time.time()

            if rerun_rhs and "A_discrete" in simulation.solutes[0].matrices:
                simulation.create_and_assemble_rhs()
            else:
                simulation.create_and_assemble_linear_system()
            
            simulation.timings["time_assembly"] = time.time() - start_time 
           
            initial_guess = np.zeros_like(simulation.rhs["rhs_discrete"])
            # Use GMRES to solve the system of equations
            if "preconditioning_matrix_gmres" in simulation.matrices:
                gmres_start_time = time.time()
                x, info, it_count = utils.solver(
                    simulation.matrices["A_discrete"],
                    simulation.rhs["rhs_discrete"],
                    simulation.gmres_tolerance,
                    simulation.gmres_restart,
                    simulation.gmres_max_iterations,
                    initial_guess = initial_guess,
                    precond = simulation.matrices["preconditioning_matrix_gmres"]
                )
                
            else:
                gmres_start_time = time.time()
                x, info, it_count = utils.solver(
                    simulation.matrices["A_discrete"],
                    simulation.rhs["rhs_discrete"],
                    simulation.gmres_tolerance,
                    simulation.gmres_restart,
                    simulation.gmres_max_iterations,
                    initial_guess = initial_guess,
                )

            simulation.timings["time_gmres"] = time.time() - gmres_start_time

            from bempp.api.assembly.blocked_operator import (
                grid_function_list_from_coefficients,
            )          

            simulation.run_info["solver_iteration_count"] = it_count
            
            solute_start = 0
            for index, solute in enumerate(simulation.solutes):

                N_dirichl = solute.dirichl_space.global_dof_count
                N_neumann = solute.neumann_space.global_dof_count
                N_dirichl_stern = solute.stern_object.dirichl_space.global_dof_count
                N_neumann_stern = solute.stern_object.neumann_space.global_dof_count
                
                N_total = N_dirichl + N_neumann + N_dirichl_stern + N_neumann_stern
                
                x_slice = x.ravel()[solute_start:solute_start + N_total]
                
                solute_start += N_total
                
                solution = grid_function_list_from_coefficients(
                    x_slice, simulation.solutes[index].matrices["A"].domain_spaces
                )
                

                solute.results["phi"]  = solution[0]
                
                if simulation.formulation_object.invert_potential:
                    solute.results["d_phi"] = (solute.ep_stern / solute.ep_in) * solution[1] 
                else:  
                    solute.results["d_phi"] = solution[1] 
                
                solute.results["phi_stern"]  = solution[2]

                if simulation.formulation_object.invert_potential:
                    solute.results["d_phi_stern"] = (solute.ep_ex / solute.ep_stern) * solution[3] 
                else:  
                    solute.results["d_phi_stern"] = solution[3]

  
            simulation.timings["time_compute_potential"] = time.time() - start_time

    

    
"""
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
        and self.pb_formulation_preconditioning is True
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
"""

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
