import bempp.api
import time
import trimesh

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
        
        self.pb_formulation_preconditioning = True
        
        if formulation=="direct" or formulation=="direct_stern" or formulation=="slic":
            self.pb_formulation_preconditioning_type = "block_diagonal"
        else:
            self.pb_formulation_preconditioning_type = "mass_matrix"
        
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
        # reset solute
        if len(self.solutes)>0:
            for index, solute in enumerate(self.solutes):
                solute.pb_formulation = self.pb_formulation
                   
    @property
    def pb_formulation_preconditioning(self):
        return self._pb_formulation_preconditioning

    @pb_formulation_preconditioning.setter
    def pb_formulation_preconditioning(self, value):
        self._pb_formulation_preconditioning = value
        # reset solute
        if len(self.solutes)>0:
            for index, solute in enumerate(self.solutes):
                solute.pb_formulation_preconditioning = self.pb_formulation_preconditioning

    @property
    def pb_formulation_preconditioning_type(self):
        return self._pb_formulation_preconditioning_type

    @pb_formulation_preconditioning_type.setter
    def pb_formulation_preconditioning_type(self, value):
        self._pb_formulation_preconditioning_type = value
        # reset solute
        if len(self.solutes)>0:
            for index, solute in enumerate(self.solutes):
                solute.pb_formulation_preconditioning_type = self.pb_formulation_preconditioning_type
                
    @property
    def ep_ex(self):
        return self._ep_ex

    @ep_ex.setter
    def ep_ex(self, value):
        self._ep_ex = value
        # reset solute
        if len(self.solutes)>0:
            for index, solute in enumerate(self.solutes):
                solute.ep_ex = self.ep_ex 
                
    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, value):
        self._kappa = value
        # reset solute
        if len(self.solutes)>0:
            for index, solute in enumerate(self.solutes):
                solute.kappa = self.kappa 
                
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
                solute.pb_formulation_preconditioning = self.pb_formulation_preconditioning 
                solute.pb_formulation_preconditioning_type = self.pb_formulation_preconditioning_type 
                self.solutes.append(solute)
        else:
            raise ValueError("Given object is not of the 'Solute' class.")

    def create_and_assemble_linear_system(self):
        from scipy.sparse import bmat, dok_matrix
        from scipy.sparse.linalg import aslinearoperator
        
        solute_count = len(self.solutes)
        #A = bempp.api.BlockedDiscreteOperator(solute_count * 2, solute_count * 2)
        
        A = np.empty((solute_count , solute_count), dtype="O")
        
        precond_matrix = []
                
        rhs_final_discrete = []
        
        # Get self interactions of each solute
        for index, solute in enumerate(self.solutes):
            solute.pb_formulation = self.pb_formulation

            solute.initialise_matrices()
            solute.initialise_rhs()
            solute.apply_preconditioning()

            #A[index * 2, index * 2] = solute.matrices["A_discrete"][0, 0]
            #A[(index * 2) + 1, index * 2] = solute.matrices["A_discrete"][1, 0]
            #A[index * 2, (index * 2) + 1] = solute.matrices["A_discrete"][0, 1]
            #A[(index * 2) + 1, (index * 2) + 1] = solute.matrices["A_discrete"][1, 1]
            
            A[index, index] = solute.matrices["A_discrete"]
            

            self.rhs["rhs_" + str(index + 1)] = [
                solute.rhs["rhs_1"],
                solute.rhs["rhs_2"],
            ]
            
            rhs_final_discrete.extend(solute.rhs["rhs_discrete"])
                        
            if solute.matrices["preconditioning_matrix_gmres"] is not None :
                
                if solute.stern_object is None:
                    precond_matrix_top_row = []
                    precond_matrix_bottom_row = []

                    for index_source, solute_source in enumerate(self.solutes):

                        if index_source == index:
                            precond_matrix_top_row.extend(solute.matrices["preconditioning_matrix_gmres"][0])
                            precond_matrix_bottom_row.extend(solute.matrices["preconditioning_matrix_gmres"][1])
                        else:
                            M = solute.dirichl_space.grid_dof_count
                            N = solute_source.dirichl_space.grid_dof_count
                            zero_matrix = dok_matrix((M,N))
                            precond_matrix_top_row.extend([zero_matrix,zero_matrix])
                            precond_matrix_bottom_row.extend([zero_matrix,zero_matrix])

                    precond_matrix.extend([precond_matrix_top_row,precond_matrix_bottom_row])
                    
                else:
                    precond_matrix_row_0 = []
                    precond_matrix_row_1 = []
                    precond_matrix_row_2 = []
                    precond_matrix_row_3 = []
                    
                    
                    for index_source, solute_source in enumerate(self.solutes):
                        
                        if index_source == index:
                            precond_matrix_row_0.extend(solute.matrices["preconditioning_matrix_gmres"][0])
                            precond_matrix_row_1.extend(solute.matrices["preconditioning_matrix_gmres"][1])
                            precond_matrix_row_2.extend(solute.matrices["preconditioning_matrix_gmres"][2])
                            precond_matrix_row_3.extend(solute.matrices["preconditioning_matrix_gmres"][3])
                        else:
                            M = solute.stern_object.dirichl_space.grid_dof_count
                            N = solute_source.stern_object.dirichl_space.grid_dof_count
                            zero_matrix = dok_matrix((M,N))
                            precond_matrix_row_0.extend([zero_matrix,zero_matrix])
                            precond_matrix_row_1.extend([zero_matrix,zero_matrix])
                            precond_matrix_row_2.extend([zero_matrix,zero_matrix])
                            precond_matrix_row_3.extend([zero_matrix,zero_matrix])
                
        if len(precond_matrix) > 0:
            precond_matrix_full = bmat(precond_matrix).tocsr()
            self.matrices["preconditioning_matrix_gmres"] = aslinearoperator(precond_matrix_full)
                

        # Calculate matrix elements for interactions between solutes

        for index_target, solute_target in enumerate(self.solutes):
            i = index_target
            for index_source, solute_source in enumerate(self.solutes):
                j = index_source

                if i!=j:

                    A_inter = self.formulation_object.lhs_inter_solute_interactions(
                        self, solute_target, solute_source        
                    )
                    
                     
                    A[i,j] = A_inter
                    #A[i    , j    ] = A_inter[0,0]
                    #A[i    , j + 1] = A_inter[0,1]
                    #A[i + 1, j    ] = A_inter[1,0]
                    #A[i + 1, j + 1] = A_inter[1,1]

        #self.matrices["A"] = A
        A_discrete = bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(A)
        self.matrices["A_discrete"] = A_discrete
        self.rhs["rhs_discrete"] = rhs_final_discrete
        
    def create_and_assemble_rhs(self):
        from scipy.sparse import bmat, dok_matrix
        from scipy.sparse.linalg import aslinearoperator
        
        solute_count = len(self.solutes)
          
        rhs_final_discrete = []
        
        for index, solute in enumerate(self.solutes):
            solute.pb_formulation = self.pb_formulation

            solute.initialise_rhs()
            solute.apply_preconditioning_rhs()

            self.rhs["rhs_" + str(index + 1)] = [
                solute.rhs["rhs_1"],
                solute.rhs["rhs_2"],
            ]
            
            rhs_final_discrete.extend(solute.rhs["rhs_discrete"])

        self.rhs["rhs_discrete"] = rhs_final_discrete

    def create_and_assemble_rhs_induced_dipole(self):
        
        rhs_final_discrete = []
        
        for index, solute in enumerate(self.solutes):
            
            solute.initialise_rhs_induced_dipole()          
            solute.apply_preconditioning_rhs()            

            self.rhs["rhs_" + str(index + 1)] = [
                solute.rhs["rhs_1"],
                solute.rhs["rhs_2"],
            ]
            
            rhs_final_discrete.extend(solute.rhs["rhs_discrete"])

        self.rhs["rhs_discrete"] = rhs_final_discrete
        
    
    def calculate_potentials(self, rerun_all=False, rerun_rhs=False):

        if rerun_all and rerun_rhs: # if both are True, just rerun_all
            rerun_rhs=False
            
        if self.solutes[0].force_field == "amoeba":
            self.calculate_potentials_polarizable(rerun_all=rerun_all, rerun_rhs=rerun_rhs)

        elif ("phi" not in self.solutes[0].results) or (rerun_all) or (rerun_rhs):
            start_time = time.time()

            if rerun_rhs and "A_discrete" in self.solutes[0].matrices:
                self.create_and_assemble_rhs()
            else:
                self.create_and_assemble_linear_system()
            
            self.timings["time_assembly"] = time.time() - start_time 
           
            initial_guess = np.zeros_like(self.rhs["rhs_discrete"])

            # Use GMRES to solve the system of equations
            if "preconditioning_matrix_gmres" in self.matrices:
                gmres_start_time = time.time()
                x, info, it_count = utils.solver(
                    self.matrices["A_discrete"],
                    self.rhs["rhs_discrete"],
                    self.gmres_tolerance,
                    self.gmres_restart,
                    self.gmres_max_iterations,
                    initial_guess = initial_guess,
                    precond = self.matrices["preconditioning_matrix_gmres"]
                )
                
            else:
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

            from bempp.api.assembly.blocked_operator import (
                grid_function_list_from_coefficients,
            )          

            self.run_info["solver_iteration_count"] = it_count
            
            solute_start = 0
            for index, solute in enumerate(self.solutes):

                N_dirichl = solute.dirichl_space.global_dof_count
                N_neumann = solute.neumann_space.global_dof_count
                N_total = N_dirichl + N_neumann
                
                x_slice = x.ravel()[solute_start:solute_start + N_total]
                
                solute_start += N_total
                
                solution = grid_function_list_from_coefficients(
                    x_slice, self.solutes[index].matrices["A"].domain_spaces
                )

                solute.results["phi"]  = solution[0]
                
                if self.formulation_object.invert_potential:
                    solute.results["d_phi"] = (solute.ep_ex / solute.ep_in) * solution[1] 
                else:  
                    solute.results["d_phi"] = solution[1] 
                  
                if solute.stern_object is not None:
                    N_dirichl = solute.stern_object.dirichl_space.global_dof_count
                    N_neumann = solute.stern_object.neumann_space.global_dof_count
                    N_total = N_dirichl + N_neumann
                    
                    x_slice = x.ravel()[solute_start:solute_start + N_total]
                
                    solute_start += N_total
                
                    solution = grid_function_list_from_coefficients(
                        x_slice, self.solutes[index].matrices["A"].domain_spaces
                    )

                    solute.results["phi_stern"]  = solution[0]
                
                    if self.formulation_object.invert_potential:
                        solute.results["d_phi_stern"] = (solute.ep_ex / solute.ep_in) * solution[1] 
                    else:  
                        solute.results["d_phi_stern"] = solution[1]

  
            self.timings["time_compute_potential"] = time.time() - start_time

    def calculate_potentials_polarizable(self, rerun_all=False, rerun_rhs=False):

        start_time = time.time()

        for index, solute in enumerate(self.solutes):
            solute.results["induced_dipole"] = np.zeros_like(solute.d)

        if rerun_rhs and "A_discrete" in self.solutes[0].matrices:
            self.create_and_assemble_rhs()
        else:
            self.create_and_assemble_linear_system()
      
        self.timings["time_assembly"] = time.time() - start_time

        induced_dipole_residual = 1.

        dipole_diff = np.zeros(len(self.solutes))

        dipole_iter_count = 0

        initial_guess = np.zeros_like(self.rhs["rhs_discrete"])
        
        self.timings["time_calc_gradient"]       = 0.
        self.timings["time_calc_induced_diss"]   = 0.
        self.timings["time_gmres"]               = 0.
        self.timings["time_assembly_rhs_induced_dipole"] = 0.
        
        while induced_dipole_residual > self.induced_dipole_iter_tol:

            start_time_rhs = time.time()
            if dipole_iter_count != 0:
                self.create_and_assemble_rhs_induced_dipole()                
            self.timings["time_assembly_rhs_induced_dipole"] += time.time() - start_time_rhs
            
            # Use GMRES to solve the system of equations
            if "preconditioning_matrix_gmres" in self.matrices:
                gmres_start_time = time.time()
                x, info, it_count = utils.solver(
                    self.matrices["A_discrete"],
                    self.rhs["rhs_discrete"],
                    self.gmres_tolerance,
                    self.gmres_restart,
                    self.gmres_max_iterations,
                    initial_guess = initial_guess,
                    precond = self.matrices["preconditioning_matrix_gmres"]
                )
                
            else:
                gmres_start_time = time.time()
                x, info, it_count = utils.solver(
                    self.matrices["A_discrete"],
                    self.rhs["rhs_discrete"],
                    self.gmres_tolerance,
                    self.gmres_restart,
                    self.gmres_max_iterations,
                    initial_guess = initial_guess,
                )

            self.timings["time_gmres"] += time.time() - gmres_start_time
            
            initial_guess = x.copy()

            from bempp.api.assembly.blocked_operator import (
                grid_function_list_from_coefficients,
            )

            solute_start = 0
            for index, solute in enumerate(self.solutes):

                N_dirichl = solute.dirichl_space.global_dof_count
                N_neumann = solute.neumann_space.global_dof_count
                N_total = N_dirichl + N_neumann
                
                x_slice = x.ravel()[solute_start:solute_start + N_total]
                
                solute_start += N_total
                
                solution = grid_function_list_from_coefficients(
                    x_slice, self.solutes[index].matrices["A"].domain_spaces
                )

                solute.results["phi"]  = solution[0]
                
                if self.formulation_object.invert_potential:
                    solute.results["d_phi"] = (solute.ep_ex / solute.ep_in) * solution[1] 
                else:  
                    solute.results["d_phi"] = solution[1] 
                    
                time_start_grad = time.time()
                solute.calculate_gradient_field()
                self.timings["time_calc_gradient"] += time.time() - time_start_grad

                d_induced_prev = solute.results["induced_dipole"].copy()
                
                time_start_induced = time.time()
                solute.calculate_induced_dipole_dissolved()
                self.timings["time_calc_induced_diss"] += time.time() - time_start_induced
                
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

    
    def calculate_solvation_energy(self, rerun_all=False, rerun_rhs=False):

        if rerun_all:
            self.calculate_potentials(rerun_all=rerun_all)
        
        if rerun_rhs:
            self.calculate_potentials(rerun_rhs=rerun_rhs)

        if "phi" not in self.solutes[0].results:
            # If surface potential has not been calculated, calculate it now
            self.calculate_potentials()
        
    
        start_time = time.time()
        for index, solute in enumerate(self.solutes):
            
            if self.solutes[0].force_field == "amoeba":
                solute.calculate_solvation_energy_polarizable()

            else:
                solute.calculate_solvation_energy()
                
                
        self.timings["time_calc_energy"] = time.time() - start_time


    def calculate_solvation_forces(self, h=0.001, rerun_all=False, force_formulation='maxwell_tensor', fdb_approx=False):

        if "phi" not in self.solutes[0].results:
            # If surface potential has not been calculated, calculate it now
            self.calculate_potentials()
        
        start_time = time.time()
        for index, solute in enumerate(self.solutes):
            solute.calculate_solvation_forces(h=h, force_formulation=force_formulation, fdb_approx=fdb_approx)

        self.timings["time_calc_force"] = time.time() - start_time

        
    def calculate_potential_solvent(self, eval_points, rerun_all=False, rerun_rhs=False):
        """
        Evaluates the potential on a cloud of points in the solvent. Needs check for multiple molecules.
        Inputs:
        -------
        eval_points: (3xN array) with 3D position of N points. 
                     If point lies in a solute it is masked out.
                 
        Outputs:
        --------
        phi_solvent: (array) electrostatic potential at eval_points
        """
        if rerun_all:
            self.calculate_potentials(rerun_all=rerun_all)
        
        if rerun_rhs:
            self.calculate_potentials(rerun_rhs=rerun_rhs)

        if "phi" not in self.solutes[0].results:
            # If surface potential has not been calculated, calculate it now
            self.calculate_potentials()
     
        # Mask out points in solute
        points_solvent = np.ones(np.shape(eval_points)[1], dtype=bool)
        for index, solute in enumerate(self.solutes):
            
            # Check if evaluation points are inside a solute
            verts = np.transpose(solute.mesh.vertices)
            faces = np.transpose(solute.mesh.elements)
            
            mesh_tri = trimesh.Trimesh(vertices = verts, faces = faces)
            
            points_solute = mesh_tri.contains(np.transpose(eval_points))
            
            points_solvent = np.logical_and(points_solvent, np.logical_not(points_solute))
            
            
        # Compute potential
        phi_solvent = np.zeros(np.shape(eval_points)[1], dtype=float)
        for index, solute in enumerate(self.solutes):
            
            
            if self.kappa < 1e-12:
                V = bempp.api.operators.potential.laplace.single_layer \
                                (solute.neumann_space, eval_points[:,points_solvent])
                K = bempp.api.operators.potential.laplace.double_layer \
                                (solute.dirichl_space, eval_points[:,points_solvent])
            else:
                V = bempp.api.operators.potential.modified_helmholtz.single_layer \
                                (solute.neumann_space, eval_points[:,points_solvent], self.kappa)
                K = bempp.api.operators.potential.modified_helmholtz.double_layer \
                                (solute.dirichl_space, eval_points[:,points_solvent], self.kappa) 

            phi_aux = K*solute.results["phi"] \
                                        - solute.ep_in/solute.ep_ex * V*solute.results["d_phi"]
            phi_solvent[points_solvent] = phi_aux[0,:]
                
        return phi_solvent
                
            
            
