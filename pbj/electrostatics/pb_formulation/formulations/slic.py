import numpy as np
import bempp.api
from bempp.api.operators.boundary import sparse, laplace
from bempp.api.linalg.iterative_solvers import IterationCounter
from .common import calculate_potential_stern
import pbj
import time
import pbj.electrostatics.utils as utils
from pbj.electrostatics.utils import matrix_to_discrete_form
import scipy.sparse.linalg



invert_potential = False


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
    
def lhs_inter_solute_interactions(self, solute_target, solute_source):
    pbj.electrostatics.pb_formulation.formulations.direct_stern.lhs_inter_solute_interactions(
        self, solute_target, solute_source
    )
    
def create_ehat_diel(self):
    dirichl_space_diel = self.dirichl_space
    neumann_space_diel = self.neumann_space
    operator_assembler = self.operator_assembler

    alpha = self.slic_alpha
    beta = self.slic_beta
    gamma = self.slic_gamma

    q = self.q
    x_q = self.x_q
    ep_in = self.ep_in
    ep_stern = getattr(self, "ep_stern", self.ep_ex)
    self.ep_stern = ep_stern

    dlp_adj_in = bempp.api.operators.boundary.laplace.adjoint_double_layer(
        dirichl_space_diel,
        dirichl_space_diel,
        neumann_space_diel,
        assembler=operator_assembler,
    )

    @bempp.api.real_callable
    def en_function(x, n, domain_index, result):
        nrm = np.sqrt(
            (x[0] - x_q[:, 0]) ** 2 + (x[1] - x_q[:, 1]) ** 2 + (x[2] - x_q[:, 2]) ** 2
        )
        result[:] = np.sum(
            np.dot(n, (np.transpose(x - x_q) / (4 * np.pi * nrm**3)) * (q))
        )

    electricfield = bempp.api.GridFunction(dirichl_space_diel, fun=en_function)

    Ksigma = dlp_adj_in * self.slic_sigma
    en = electricfield - Ksigma
    mu = -alpha * np.tanh(-gamma)
    h = alpha * np.tanh(beta * en.coefficients - gamma) + mu
    f = ep_in / (ep_stern - ep_in) - h
    f_div = f / (1 + f)
    f_fun = bempp.api.GridFunction(neumann_space_diel, coefficients=f_div)
    self.slic_e_hat_diel = bempp.api.assembly.boundary_operator.MultiplicationOperator(
        f_fun, neumann_space_diel, neumann_space_diel, neumann_space_diel
    )

def solve_sigma(self):
    dirichl_space_diel = self.dirichl_space
    neumann_space_diel = self.neumann_space
    operator_assembler = self.operator_assembler

    identity_diel = sparse.identity(
        dirichl_space_diel, dirichl_space_diel, dirichl_space_diel
    )
    slp_in_diel = laplace.single_layer(
        neumann_space_diel,
        dirichl_space_diel,
        dirichl_space_diel,
        assembler=operator_assembler,
    )
    dlp_in_diel = laplace.double_layer(
        dirichl_space_diel,
        dirichl_space_diel,
        dirichl_space_diel,
        assembler=operator_assembler,
    )

    rhs_sigma = (
        slp_in_diel * self.results["d_phi"]
        - (-0.5 * identity_diel + dlp_in_diel) * self.results["phi"]
    )
    """
    sigma, _ = bempp.api.linalg.gmres(
        slp_in_diel,
        rhs_sigma,
        tol=self.gmres_tolerance,
        maxiter=self.gmres_max_iterations,
        use_strong_form=True
    )
    """
    
    if not slp_in_diel.range.is_compatible(rhs_sigma.space):
        raise ValueError(
            "The range of A and the domain of A must have"
            + "the same number of unknowns if the strong form is used."
        )
        
    A_op = slp_in_diel.strong_form()
    b_vec = rhs_sigma.coefficients
    
    callback = IterationCounter(False)

    bempp.api.log("PBJ: Starting GMRES iterations for sigma (SLIC)")

    start_time = time.time()
    x, info = scipy.sparse.linalg.gmres(
        A_op, b_vec, x0 = self.slic_sigma.coefficients, 
        tol=self.gmres_tolerance, maxiter=self.gmres_max_iterations, callback=callback
    )
    end_time = time.time()
    bempp.api.log(
        "GMRES finished in %i iterations and took %.2E sec."
        % (callback.count, end_time - start_time)
    )
    
    sigma = bempp.api.GridFunction(
        slp_in_diel.domain, coefficients=x.ravel()
    )

    self.slic_sigma = sigma


def calculate_potential(simulation, rerun_all, rerun_rhs):
    
    for index, solute in enumerate(simulation.solutes):
        
        dirichl_space_diel = solute.dirichl_space

        ep_stern = getattr(solute, "ep_stern", simulation.ep_ex)
        solute.ep_stern = ep_stern

        if solute.stern_object is None:
            pbj.electrostatics.pb_formulation.formulations.direct_stern.create_stern_mesh(
                solute
            )

        max_iterations = simulation.slic_max_iterations

        solute.slic_sigma = bempp.api.GridFunction(
            dirichl_space_diel, coefficients=np.zeros(dirichl_space_diel.global_dof_count)
        )
        
        #solute.initialise_rhs()

    simulation.timings["time_gmres"] = []
    simulation.timings["time_compute_potential"] = []
    simulation.run_info["solver_iteration_count"] = []

    time_matrix_initialisation = []
    time_matrix_assembly = []
    time_preconditioning = []

    bempp.api.log("PBJ: Starting self-consistent SLIC iterations")
    # iteration 0
    calculate_potential_stern(simulation)
    
    # Cache matrices without factors for SLIC iterations
    matrix_cache = np.empty(len(simulation.solutes), dtype="O")
    inter_matrix_cache = np.empty(len(simulation.solutes), dtype="O")
    phi_old = np.array([])
    for index, solute in enumerate(simulation.solutes):
        matrix_solute = [solute.matrices["A"][1,1]*(solute.ep_stern/solute.ep_in),
                         solute.matrices["A"][2,1]*(solute.ep_stern/solute.ep_in),
                         solute.matrices["A"][3,3]*(solute.ep_ex/solute.ep_stern)
                        ]
        matrix_cache[index] = matrix_solute
        
        inter_matrix_solute = []
        for index_j, solute_j in enumerate(simulation.solutes):
            if index_j != index:
                if index>index_j:
                    index_array = index_j
                else:
                    index_array = index_j - 1
                
                inter_matrix_solute.append(solute.matrices["A_inter"][index_array][3,3]*(solute.ep_ex/solute.ep_stern))
        
        inter_matrix_cache[index] = inter_matrix_solute
        
        phi_old = np.append(phi_old, solute.results["phi"].coefficients.copy())            
    
    it = 1
    phi_L2error = 1.
    tolerance = simulation.slic_tolerance
    solute_count = len(simulation.solutes)
    while it < max_iterations and phi_L2error > tolerance:
    
        A = np.empty((solute_count , solute_count), dtype="O")
        
        #update e_hats
        for index, solute in enumerate(simulation.solutes):
            qtotal = np.sum(solute.q)
            d1 = -qtotal / solute.ep_stern
            solve_sigma(solute)
            create_ehat_diel(solute)
            d2 = solute.results["d_phi_stern"].integrate()[0]
            solute.slic_e_hat_stern = d1 / d2
            
        for index, solute in enumerate(simulation.solutes):

            A_solute = bempp.api.BlockedOperator(4, 4)
           
            A_solute[0,0] = solute.matrices["A"][0,0]
            A_solute[0,1] = solute.matrices["A"][0,1]
            A_solute[0,2] = solute.matrices["A"][0,2]
            A_solute[0,3] = solute.matrices["A"][0,3]
            A_solute[1,0] = solute.matrices["A"][1,0]
            A_solute[1,1] = matrix_cache[index][0]*solute.slic_e_hat_diel
            A_solute[1,2] = solute.matrices["A"][1,2]
            A_solute[1,3] = solute.matrices["A"][1,3]
            A_solute[2,0] = solute.matrices["A"][2,0]
            A_solute[2,1] = matrix_cache[index][1]*solute.slic_e_hat_diel
            A_solute[2,2] = solute.matrices["A"][2,2]
            A_solute[2,3] = solute.matrices["A"][2,3]
            A_solute[3,0] = solute.matrices["A"][3,0]
            A_solute[3,1] = solute.matrices["A"][3,0]
            A_solute[3,2] = solute.matrices["A"][3,2]
            A_solute[3,3] = matrix_cache[index][2]*solute.slic_e_hat_stern
            
            solute.matrices["A"] = A_solute
                       
            solute.matrices["A_discrete"] = solute.matrices["A"].weak_form()
            
            A[index, index] = solute.matrices["A_discrete"]
            
            for index_j, solute_j in enumerate(simulation.solutes):
                
                if index_j != index:
                    A_inter = bempp.api.BlockedOperator(4, 4)
                    
                    if index>index_j:
                        index_array = index_j
                    else:
                        index_array = index_j - 1
                        
                    A_inter[0,0] = solute.matrices["A_inter"][index_array][0,0]
                    A_inter[0,1] = solute.matrices["A_inter"][index_array][0,1]
                    A_inter[0,2] = solute.matrices["A_inter"][index_array][0,2]
                    A_inter[0,3] = solute.matrices["A_inter"][index_array][0,3]
                    A_inter[1,0] = solute.matrices["A_inter"][index_array][1,0]
                    A_inter[1,1] = solute.matrices["A_inter"][index_array][1,1]
                    A_inter[1,2] = solute.matrices["A_inter"][index_array][1,2]
                    A_inter[1,3] = solute.matrices["A_inter"][index_array][1,3]
                    A_inter[2,0] = solute.matrices["A_inter"][index_array][2,0]
                    A_inter[2,1] = solute.matrices["A_inter"][index_array][2,1]
                    A_inter[2,2] = solute.matrices["A_inter"][index_array][2,2]
                    A_inter[2,3] = solute.matrices["A_inter"][index_array][2,3]
                    A_inter[3,0] = solute.matrices["A_inter"][index_array][3,0]
                    A_inter[3,1] = solute.matrices["A_inter"][index_array][3,0]
                    A_inter[3,2] = solute.matrices["A_inter"][index_array][3,2]                
                    A_inter[3,3] = inter_matrix_cache[index][index_array]*solute_j.slic_e_hat_stern
                
                    solute.matrices["A_inter"][index_array] = A_inter
                    
                    A[index,index_j] = A_inter.weak_form()
                
        A_discrete = bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(A)
        simulation.matrices["A_discrete"] = A_discrete
            
        #update_and_assemble_linear_system_slic(simulation, matrix_cache)
        
        calculate_potential_slic(simulation)
        
        phi_new = np.array([])
        for index, solute in enumerate(simulation.solutes):
            phi_new = np.append(phi_new, solute.results["phi"].coefficients.copy())
        
        phi_L2error = np.sqrt(
            np.sum((phi_old - phi_new) ** 2)
            / np.sum(phi_new ** 2)
        )

        phi_old = phi_new.copy()
        

        bempp.api.log("PBJ: Self-consistent iteration %i, residual %e"%(it,phi_L2error))
        
        it += 1

        for index, solute in enumerate(simulation.solutes):
            time_matrix_initialisation.append(solute.timings["time_matrix_initialisation"])
            time_matrix_assembly.append(simulation.timings["time_assembly"])
            time_preconditioning.append(solute.timings["time_preconditioning"])

    for index, solute in enumerate(simulation.solutes):
        solute.timings["time_matrix_initialisation"] = time_matrix_initialisation
        simulation.timings["time_assembly"] = time_matrix_assembly
        solute.timings["time_preconditioning"] = time_preconditioning

    
    
def calculate_potential_slic(simulation):
    start_time = time.time()

    simulation.timings["time_assembly"] = time.time() - start_time 

    initial_guess = np.zeros_like(simulation.rhs["rhs_discrete"])
    
    i = 0
    for index, solute in enumerate(simulation.solutes):
        N_dirichl = solute.dirichl_space.global_dof_count
        N_neumann = solute.neumann_space.global_dof_count
        
        initial_guess[i:i+N_dirichl] = solute.results["phi"].coefficients
        i += N_dirichl
        initial_guess[i:i+N_neumann] = solute.results["d_phi"].coefficients
        i += N_neumann
        
        N_dirichl = solute.stern_object.dirichl_space.global_dof_count
        N_neumann = solute.stern_object.neumann_space.global_dof_count
        
        initial_guess[i:i+N_dirichl] = solute.results["phi_stern"].coefficients
        i += N_dirichl
        initial_guess[i:i+N_neumann] = solute.results["d_phi_stern"].coefficients
        i += N_neumann
    
    bempp.api.log("PBJ: Start GMRES iterations for surface potential")
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

def update_and_assemble_linear_system_slic(simulation, matrix_cache):
    from scipy.sparse import bmat, dok_matrix
    from scipy.sparse.linalg import aslinearoperator

    solute_count = len(simulation.solutes)
    
    A = np.empty((solute_count , solute_count), dtype="O")
    
    for index, solute in enumerate(simulation.solutes):
        A[index, index] = solute.matrices["A_discrete"]
  
    # Calculate matrix elements for interactions between solutes
    for index_target, solute_target in enumerate(simulation.solutes):
        i = index_target
        for index_source, solute_source in enumerate(simulation.solutes):
            j = index_source

            if i!=j:
                if i>j:
                    index_array = j
                else:
                    index_array = j - 1
                    
                A[i,j] = solute.matrices["A_inter"][index_array]

    #simulation.matrices["A"] = A
    A_discrete = bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(A)
    simulation.matrices["A_discrete"] = A_discrete
