import bempp.api
import time

import numpy as np
import pbj.electrostatics.solute
import pbj.electrostatics.pb_formulation.formulations as pb_formulations
import pbj.electrostatics.utils as utils


class Simulation:
    def __init__(self, formulation="direct", print_times=False):

        self._pb_formulation = formulation
        self.formulation_object = getattr(pb_formulations, self.pb_formulation, None)
        if self.formulation_object is None:
            raise ValueError("Unrecognised formulation type %s" % self.pb_formulation)

        self.solvent_parameters = dict()
        self.solvent_parameters["ep"] = 80.0

        self.gmres_tolerance = 1e-5
        self.gmres_restart = 1000
        self.gmres_max_iterations = 1000

        self.induced_dipole_iter_tol = 1e-2

        self.solutes = list()
        self.matrices = dict()
        self.rhs = dict()
        self.timings = dict()
        self.run_info = dict()
        
        self.ep_ex = 80.0
        self.kappa = 0.125
        
        self.operator_assembler = "dense"

        self.SOR = 0.7

    @property
    def pb_formulation(self):
        return self._pb_formulation

    @pb_formulation.setter
    def pb_formulation(self, value):
        self._pb_formulation = value
        self.formulation_object = getattr(pb_formulations, self.pb_formulation, None)
        self.matrices["preconditioning_matrix_gmres"] = None
        if self.formulation_object is None:
            raise ValueError("Unrecognised formulation type %s" % self.pb_formulation)

    def add_solute(self, solute):

        if isinstance(solute, pbj.electrostatics.solute.Solute):
            if solute in self.solutes:
                print(
                    "Solute object is already added to this simulation. Ignoring this add command."
                )
            else:
                solute.ep_ex = self.ep_ex
                solute.kappa = self.kappa
                solute.SOR = self.SOR
                solute.induced_dipole_iter_tol = self.induced_dipole_iter_tol
                solute.operator_assembler = self.operator_assembler
                self.solutes.append(solute)
        else:
            raise ValueError("Given object is not of the 'Solute' class.")

    def create_and_assemble_linear_system(self):
        surface_count = len(self.solutes)
        A = bempp.api.BlockedOperator(surface_count * 2, surface_count * 2)

        # Get self interactions of each solute
        for index, solute in enumerate(self.solutes):
            solute.pb_formulation = self.pb_formulation
            #solute.pb_formulation_preconditioning = False


            solute.initialise_matrices()
            solute.initialise_rhs()
            solute.assemble_matrices()
            solute.apply_preconditioning()

            A[index * 2, index * 2] = solute.matrices["A"][0, 0]
            A[(index * 2) + 1, index * 2] = solute.matrices["A"][1, 0]
            A[index * 2, (index * 2) + 1] = solute.matrices["A"][0, 1]
            A[(index * 2) + 1, (index * 2) + 1] = solute.matrices["A"][1, 1]

            self.rhs["rhs_" + str(index + 1)] = [
                solute.rhs["rhs_1"],
                solute.rhs["rhs_2"],
            ]

        # Calculate matrix elements for interactions between solutes

        for index_target, solute_target in enumerate(self.solutes):
            i = index_target*2
            for index_source, solute_source in enumerate(self.solutes):
                j = index_source*2

                if i!=j:

                    A_inter = self.formulation_object.lhs_inter_solute_interactions(
                        self, solute_target, solute_source        
                    )
                    
                    A[i    , j    ] = A_inter[0,0]
                    A[i    , j + 1] = A_inter[0,1]
                    A[i + 1, j    ] = A_inter[1,0]
                    A[i + 1, j + 1] = A_inter[1,1]

        self.matrices["A"] = A

    def create_and_assemble_rhs(self):

        for index, solute in enumerate(self.solutes):

            self.rhs["rhs_" + str(index + 1)] = [
                solute.rhs["rhs_1"],
                solute.rhs["rhs_2"],
            ]


    def apply_preconditioning(self):
        self.matrices["A_final"] = self.matrices["A"]

        rhs_final = []
        count = 0
        for key, solute_rhs in self.rhs.items():
            if count >= len(self.matrices["A"].domain_spaces) / 2:
                break
            else:
                rhs_final.extend(solute_rhs)
                count += 1

        self.rhs["rhs_final"] = rhs_final

        self.matrices["A_discrete"] = utils.matrix_to_discrete_form(
            self.matrices["A_final"], "weak"
        )
        self.rhs["rhs_discrete"] = utils.rhs_to_discrete_form(
            self.rhs["rhs_final"], "weak", self.matrices["A"]
        )

    def apply_preconditioning_rhs(self):

        rhs_final = []
        count = 0
        for key, solute_rhs in self.rhs.items():
            if count >= len(self.matrices["A"].domain_spaces) / 2:
                break
            else:
                rhs_final.extend(solute_rhs)
                count += 1

        self.rhs["rhs_final"] = rhs_final

        self.rhs["rhs_discrete"] = utils.rhs_to_discrete_form(
            self.rhs["rhs_final"], "weak", self.matrices["A"]
        )
        
    
    def calculate_potentials(self):

        if self.solutes[0].force_field == "amoeba":
            self.calculate_potentials_polarizable()

        else:
            start_time = time.time()

            self.create_and_assemble_linear_system()
            self.apply_preconditioning()

            # Use GMRES to solve the system of equations
            gmres_start_time = time.time()
            x, info, it_count = utils.solver(
                self.matrices["A_discrete"],
                self.rhs["rhs_discrete"],
                self.gmres_tolerance,
                self.gmres_restart,
                self.gmres_max_iterations,
            )

            self.timings["time_gmres"] = time.time() - gmres_start_time

            from bempp.api.assembly.blocked_operator import (
                grid_function_list_from_coefficients,
            )

            solution = grid_function_list_from_coefficients(
                x.ravel(), self.matrices["A"].domain_spaces
            )

            self.run_info["solver_iteration_count"] = it_count
            for index, solute in enumerate(self.solutes):

                N_dirichl = solute.dirichl_space.global_dof_count
                N_neumann = solute.neumann_space.global_dof_count

                solute.results["phi"]  = solution[2*index]
                
                if self.formulation_object.invert_potential:
                    solute.results["d_phi"] = (solute.ep_ex / solute.ep_in) * solution[2*index+1] 
                else:  
                    solute.results["d_phi"] = solution[2*index+1] 

            self.timings["time_compute_potential"] = time.time() - start_time

    def calculate_potentials_polarizable(self):

        start_time = time.time()

        for index, solute in enumerate(self.solutes):
            solute.results["induced_dipole"] = np.zeros_like(solute.d)

        self.create_and_assemble_linear_system()
        self.apply_preconditioning()

        induced_dipole_residual = 1.

        dipole_diff = np.zeros(len(self.solutes))

        dipole_iter_count = 0
        #self.results["dipole_iter_count"] = 0 # find better place to store this


        initial_guess = np.zeros_like(self.rhs["rhs_discrete"])
        
        while induced_dipole_residual > self.induced_dipole_iter_tol:

            if dipole_iter_count != 0:
                self.create_and_assemble_rhs()
                self.apply_preconditioning_rhs()
                

            # Use GMRES to solve the system of equations
            gmres_start_time = time.time()
            x, info, it_count = utils.solver(
                self.matrices["A_discrete"],
                self.rhs["rhs_discrete"],
                self.gmres_tolerance,
                self.gmres_restart,
                self.gmres_max_iterations,
                initial_guess = initial_guess,
            )

            self.timings["time_gmres"] = time.time() - gmres_start_time
            
            initial_guess = x.copy()

            from bempp.api.assembly.blocked_operator import (
                grid_function_list_from_coefficients,
            )

            solution = grid_function_list_from_coefficients(
                x.ravel(), self.matrices["A"].domain_spaces
            )

            for index, solute in enumerate(self.solutes):

                N_dirichl = solute.dirichl_space.global_dof_count
                N_neumann = solute.neumann_space.global_dof_count

                solute.results["phi"]  = solution[2*index]
                
                if self.formulation_object.invert_potential:
                    solute.results["d_phi"] = (solute.ep_ex / solute.ep_in) * solution[2*index+1] 
                else:  
                    solute.results["d_phi"] = solution[2*index+1]
                    

                solute.calculate_gradient_field()

                d_induced_prev = solute.results["induced_dipole"].copy()
                
                solute.calculate_induced_dipole_dissolved()

                d_induced = solute.results["induced_dipole"]

                dipole_diff[index] = np.max(np.sqrt(np.sum(
                                (np.linalg.norm(d_induced_prev-d_induced,axis=1))**2)/len(d_induced)
                            )
                        )

            induced_dipole_residual = np.max(dipole_diff)

            print("Induced dipole iteration %i -> residual: %s"%(
                        dipole_iter_count, induced_dipole_residual
                        )
                    )

            dipole_iter_count += 1

             

        self.timings["time_compute_potential"] = time.time() - start_time



        # Print times, if this is desired
#if self.print_times:
#           show_potential_calculation_times(self)

    
    def calculate_solvation_energy(self, rerun_all=False):

        if rerun_all:
            self.calculate_potential(rerun_all)

        if "phi" not in self.solutes[0].results:
            # If surface potential has not been calculated, calculate it now
            self.calculate_potentials()
        
    
        for index, solute in enumerate(self.solutes):
            
            if self.solutes[0].force_field == "amoeba":
                solute.calculate_solvation_energy_polarizable()

            else:
                solute.calculate_solvation_energy()


    def calculate_solvation_forces(self, h=0.001, rerun_all=False):

        if "phi" not in self.solutes[0].results:
            # If surface potential has not been calculated, calculate it now
            self.calculate_potentials()
        
    
        for index, solute in enumerate(self.solutes):
            solute.calculate_solvation_forces()
