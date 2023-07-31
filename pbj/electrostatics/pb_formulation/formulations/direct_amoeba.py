import numpy as np
import bempp.api
import os
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
from numba import jit
import time
import pbj
import pbj.electrostatics.utils as utils


invert_potential = False


def verify_parameters(self):
    return True


def lhs(self):
    pbj.electrostatics.pb_formulation.formulations.direct.lhs(self)


def rhs(self):
    
    force_field = self.force_field
    dirichl_space = self.dirichl_space
    neumann_space = self.neumann_space
    q = self.q
    x_q = self.x_q
    if force_field == "amoeba":
        d = self.d
        Q = self.Q
        d_induced = self.d_induced
    ep_in = self.ep_in
    rhs_constructor = self.rhs_constructor

    if rhs_constructor == "fmm":
            
        @bempp.api.real_callable
        def multipolar_charges_fun(x, n, i, result): # not using fmm
            T2 = np.zeros((len(x_q),3,3))
            phi = 0
            dist = x - x_q
            norm = np.sqrt(np.sum((dist*dist), axis = 1))
            T0 = 1/norm[:]
            T1 = np.transpose(dist.transpose()/norm**3)
            T2[:,:,:] = np.ones((len(x_q),3,3))[:]* dist.reshape((len(x_q),1,3))* \
            np.transpose(np.ones((len(x_q),3,3))*dist.reshape((len(x_q),1,3)), (0,2,1))/norm.reshape((len(x_q),1,1))**5
            phi = np.sum(q[:]*T0[:]) + np.sum(T1[:]*(d[:])) + 0.5*np.sum(np.sum(T2[:]*Q[:],axis=1))
            result[0] = (phi/(4*np.pi*ep_in))
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

            dist = np.zeros((3,len(x_q)))
            dist[0,:] = x[0] - x_q[:, 0]
            dist[1,:] = x[1] - x_q[:, 1]
            dist[2,:] = x[2] - x_q[:, 2]

            #dist = x - x_q
            #norm = np.sqrt(np.sum((dist*dist), axis = 1))

            norm = np.sqrt(
                (dist[0,:]) ** 2
                + (dist[1,:]) ** 2
                + (dist[2,:]) ** 2
            )

            T0 = 1/norm

            T1 = np.zeros((3, len(x_q)))
            T1[0,:] = dist[0,:]#/norm**3
            T1[1,:] = dist[1,:]#/norm**3
            T1[2,:] = dist[2,:]#/norm**3

            T1 /= (norm*norm*norm)

            #T1 = np.transpose(dist.transpose()/norm**3)

            T2 = np.zeros((3,3,len(x_q)))

            T2[0,0,:] = dist[0,:] * dist[0,:]/norm**5
            T2[0,1,:] = dist[0,:] * dist[1,:]/norm**5
            T2[0,2,:] = dist[0,:] * dist[2,:]/norm**5
            T2[1,1,:] = dist[1,:] * dist[1,:]/norm**5
            T2[1,2,:] = dist[1,:] * dist[2,:]/norm**5
            T2[2,2,:] = dist[2,:] * dist[2,:]/norm**5

            T2[2,1,:] = T2[1,2,:]
            T2[1,0,:] = T2[0,1,:]
            T2[2,0,:] = T2[0,2,:]

            #T2[:,:,:] = np.ones((len(x_q),3,3))[:]* dist.reshape((len(x_q),1,3))* \
            #np.transpose(np.ones((len(x_q),3,3))*dist.reshape((len(x_q),1,3)), (0,2,1))/norm.reshape((len(x_q),1,1))**5

            phi = np.sum(q*T0) + np.sum(T1.transpose()*(d)) + 0.5*np.sum(np.sum(T2.transpose()[:]*Q[:],axis=1))
            # only computes permanent multipole. Having a initial induced component is to be implemented
            result[0] = (phi/(4*np.pi*ep_in))

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=multipolar_charges_fun)            
        rhs_2 = bempp.api.GridFunction(neumann_space, fun=zero)

    self.rhs["rhs_1"], self.rhs["rhs_2"] = rhs_1, rhs_2
    self.rhs["rhs_permanent_multipole_1"], self.rhs["rhs_permanent_multipole_2"] = rhs_1, rhs_2


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
        def dipole_charges_fun(x, n, i, result): # not using fmm
            dist = x - x_q
            norm = np.sqrt(np.sum((dist*dist), axis = 1))
            T1 = np.transpose(dist.transpose()/norm**3)
            phi = np.sum(T1[:]*d_induced[:])
            result[0] = (phi/(4*np.pi*ep_in))
            
        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=dipole_charges_fun)

        # rhs_2 = bempp.api.GridFunction(neumann_space, fun=zero)
        rhs_2 = bempp.api.GridFunction(neumann_space, coefficients=coefs)

    else:

        @bempp.api.real_callable
        def zero(x, n, domain_index, result):
            result[0] = 0
                
        @bempp.api.real_callable
        def dipole_charges_fun(x, n, i, result):

            dist = np.zeros((3,len(x_q)))
            dist[0,:] = x[0] - x_q[:, 0]
            dist[1,:] = x[1] - x_q[:, 1]
            dist[2,:] = x[2] - x_q[:, 2]

            norm = np.sqrt(
                (dist[0,:]) ** 2
                + (dist[1,:]) ** 2
                + (dist[2,:]) ** 2
            )

            T1 = np.zeros((3, len(x_q)))
            T1[0,:] = dist[0,:]
            T1[1,:] = dist[1,:]
            T1[2,:] = dist[2,:]

            T1 /= (norm*norm*norm)

            phi = np.sum(T1.transpose()*d_induced)
            result[0] = (phi/(4*np.pi*ep_in))

        rhs_1 = bempp.api.GridFunction(dirichl_space, fun=dipole_charges_fun)

        rhs_2 = bempp.api.GridFunction(neumann_space, fun=zero)

    # Add induced dipole component to already existing rhs with permanent multipoles
    self.rhs["rhs_1"] = self.rhs["rhs_permanent_multipole_1"] + rhs_1
    self.rhs["rhs_2"] = self.rhs["rhs_permanent_multipole_2"] + rhs_2
    
def block_diagonal_preconditioner(solute):
    pbj.electrostatics.pb_formulation.formulations.direct.block_diagonal_preconditioner(solute)



def mass_matrix_preconditioner(solute):
    pbj.electrostatics.pb_formulation.formulations.direct.mass_matrix_preconditioner(solute)

    
def mass_matrix_preconditioner_rhs(solute):
    pbj.electrostatics.pb_formulation.formulations.direct.mass_matrix_preconditioner_rhs(solute)


def calculate_potential(simulation, rerun_all=False, rerun_rhs=False):

    start_time = time.time()

    for index, solute in enumerate(simulation.solutes):
        solute.results["induced_dipole"] = np.zeros_like(solute.d)

    if rerun_rhs and "A_discrete" in simulation.solutes[0].matrices:
        simulation.create_and_assemble_rhs()
    else:
        simulation.create_and_assemble_linear_system()

    simulation.timings["time_assembly"] = time.time() - start_time

    induced_dipole_residual = 1.

    dipole_diff = np.zeros(len(simulation.solutes))

    dipole_iter_count = 0

    initial_guess = np.zeros_like(simulation.rhs["rhs_discrete"])

    simulation.timings["time_calc_gradient"]       = 0.
    simulation.timings["time_calc_induced_diss"]   = 0.
    simulation.timings["time_gmres"]               = 0.
    simulation.timings["time_assembly_rhs_induced_dipole"] = 0.

    bempp.api.log("PBJ: Starting self-consistent interations for induced dipole in dissolved state")
    while induced_dipole_residual > simulation.induced_dipole_iter_tol:

        start_time_rhs = time.time()
        if dipole_iter_count != 0:
            create_and_assemble_rhs_induced_dipole(simulation)                
        simulation.timings["time_assembly_rhs_induced_dipole"] += time.time() - start_time_rhs

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

        simulation.timings["time_gmres"] += time.time() - gmres_start_time

        initial_guess = x.copy()

        from bempp.api.assembly.blocked_operator import (
            grid_function_list_from_coefficients,
        )

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

            time_start_grad = time.time()
            solute.calculate_gradient_field()
            simulation.timings["time_calc_gradient"] += time.time() - time_start_grad

            d_induced_prev = solute.results["induced_dipole"].copy()

            time_start_induced = time.time()
            calculate_induced_dipole_dissolved(solute)
            simulation.timings["time_calc_induced_diss"] += time.time() - time_start_induced

            d_induced = solute.results["induced_dipole"]

            dipole_diff[index] = np.max(np.sqrt(np.sum(
                            (np.linalg.norm(d_induced_prev-d_induced,axis=1))**2)/len(d_induced)
                        )
                    )

        induced_dipole_residual = np.max(dipole_diff)

        bempp.api.log("PBJ: Dissolved induced dipole iteration %i -> residual: %s"%(
                    dipole_iter_count, induced_dipole_residual
                    )
                )

        dipole_iter_count += 1

    simulation.timings["time_compute_potential"] = time.time() - start_time
    
    

def create_and_assemble_rhs_induced_dipole(simulation):

    rhs_final_discrete = []

    for index, solute in enumerate(simulation.solutes):

        initialise_rhs_induced_dipole(solute)          
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
        dirichl_space_source, dirichl_space_target, dirichl_space_target, kappa, assembler=operator_assembler
    )
    slp = modified_helmholtz.single_layer(
        neumann_space_source, neumann_space_target, neumann_space_target, kappa,  assembler=operator_assembler
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
    A_inter[1, 0] = - dlp
    A_inter[1, 1] = (ep_in / ep_out) * slp

    solute_target.matrices["A_inter"].append(A_inter)
    
    #return A_inter.weak_form()  # should always be weak_form, as preconditioner doesn't touch it

        
def initialise_rhs_induced_dipole(self):
    start_rhs = time.time()
    # Verify if parameters are already set and then save RHS
    if self.formulation_object.verify_parameters(self):
        self.formulation_object.rhs_induced_dipole(self)
    self.timings["time_rhs_initialisation"] = time.time() - start_rhs
    
    
def calculate_solvation_energy_polarizable(solute):

    start_time = time.time()

    if "phi" not in solute.results:
        print("Please compute surface potential first with simulation.calculate_potentials()")
        return

    q = solute.q
    d = solute.d
    Q = solute.Q

    solution_dirichl = solute.results["phi"]
    solution_neumann = solute.results["d_phi"]

    from bempp.api.operators.potential.laplace import single_layer, double_layer

    slp_q = single_layer(solute.neumann_space, solute.x_q.transpose())
    dlp_q = double_layer(solute.dirichl_space, solute.x_q.transpose())
    phi_q = slp_q * solution_neumann - dlp_q * solution_dirichl

    solute.results["phir_charges"] = phi_q[0,:]

    if "gradphir_charges" not in solute.results:
        solute.calculate_gradient_field()

    solute.calculate_gradgradient_field()

    # total solvation energy applying constant to get units [kcal/mol]
    q_aux = 0
    d_aux = 0
    Q_aux = 0

    dphi_q = solute.results["gradphir_charges"]
    ddphi_q = solute.results["gradgradphir_charges"]

    for i in range(len(q)):
        q_aux += q[i]*phi_q[0,i]

        for j in range(3):
            d_aux += d[i,j]*dphi_q[i,j]

            for k in range(3):
                Q_aux += Q[i,j,k]*ddphi_q[i,j,k]/6.

    solvent_energy = 2 * np.pi * 332.064 * (q_aux + d_aux + Q_aux)
    coulomb_energy_dissolved = calculate_coulomb_energy_multipole(solute, state="dissolved")


    calculate_induced_dipole_vacuum(solute) 
    coulomb_energy_vacuum = calculate_coulomb_energy_multipole(solute, state="vacuum")

    solute.results["solvation_energy"] = solvent_energy + coulomb_energy_dissolved - coulomb_energy_vacuum
    solute.timings["time_calc_energy"] = time.time() - start_time

    if solute.print_times:
        print(
            "It took ",
            solute.timings["time_calc_energy"],
            " seconds to compute the solvation energy",
        )

        
def calculate_coulomb_energy_multipole(solute, state):
    """
    Calculates the Coulomb energy 

    state: (string) dissolved or vacuum, to choose which induced dipole to use
    """

    N = len(solute.x_q)

    q = solute.q
    d = solute.d
    Q = solute.Q

    #phi, dphi and ddphi from permanent multipoles
    phi_perm   = calculate_coulomb_phi_multipole(solute)
    flag_polar_group = False
    dphi_perm  = calculate_coulomb_dphi_multipole(solute, flag_polar_group) # Recalculate for energy as flag = False
    ddphi_perm = calculate_coulomb_ddphi_multipole(solute)

    solute.results["phi_perm_multipoles"]         = phi_perm
    solute.results["gradphi_perm_multipoles"]     = dphi_perm
    solute.results["gradgradphi_perm_multipoles"] = ddphi_perm


    #phi, dphi and ddphi from induced dipoles
    phi_thole = calculate_coulomb_phi_multipole_Thole(solute, state)
    dphi_thole = calculate_coulomb_dphi_multipole_Thole(solute, state)
    ddphi_thole = calculate_coulomb_ddphi_multipole_Thole(solute, state)

    solute.results["phi_induced_dipole_"+state]         = phi_thole
    solute.results["gradphi_induced_dipole_"+state]     = dphi_thole
    solute.results["gradgradphi_induced_dipole_"+state] = ddphi_thole

    phi   = phi_perm  + phi_thole
    dphi  = dphi_perm + dphi_thole
    ddphi = ddphi_perm + ddphi_thole

    point_energy = q[:]*phi[:] + np.sum(d[:] * dphi[:], axis = 1) + (np.sum(np.sum(Q[:]*ddphi[:], axis = 1), axis = 1))/6.

    cal2J = 4.184
    ep_vacc = 8.854187818e-12
    qe = 1.60217646e-19
    Na = 6.0221415e+23
    C0 = qe**2*Na*1e-3*1e10/(cal2J*ep_vacc)

    coulomb_energy = sum(point_energy) * 0.5*C0/(4*np.pi*solute.ep_in) 

    return coulomb_energy
    
def calculate_induced_dipole_dissolved(solute):

    N = len(solute.x_q)

    p12scale_temp = solute.p12scale
    p13scale_temp = solute.p13scale

    u12scale = 1.0
    u13scale = 1.0

    solute.p12scale = u12scale
    solute.p13scale = u13scale  # scaling for induced dipole calculation

    alphaxx = solute.alpha[:,0,0]

    if "d_phi_coulomb_multipole" not in solute.results:
        dphi_perm = calculate_coulomb_dphi_multipole(solute)
        solute.results["d_phi_coulomb_multipole"] = dphi_perm

    dphi_Thole = calculate_coulomb_dphi_multipole_Thole(solute, state="dissolved")    


    solute.p12scale = p12scale_temp
    solute.p13scale = p13scale_temp


    dphi_coul = solute.results["d_phi_coulomb_multipole"] + dphi_Thole

    dphi_reac = solute.results["gradphir_charges"] 

    d_induced = solute.results["induced_dipole"]

    SOR = solute.SOR
    for i in range(N):

        E_total = (dphi_coul[i]/solute.ep_in + 4*np.pi*dphi_reac[i])*-1
        d_induced[i] = d_induced[i]*(1 - SOR) + np.dot(alphaxx[i], E_total)*SOR

    solute.d_induced[:] = d_induced[:]

    solute.results["induced_dipole"] = d_induced


def calculate_induced_dipole_vacuum(solute):

    N = len(solute.x_q)


    u12scale = 1.0
    u13scale = 1.0


    alphaxx = solute.alpha[:,0,0]


    if "d_phi_coulomb_multipole" not in solute.results:
        dphi_perm = calculate_coulomb_dphi_multipole(solute)
        solute.results["d_phi_coulomb_multipole"] = dphi_perm

    d_induced = np.zeros_like(solute.d)
    solute.results["induced_dipole_vacuum"] = d_induced

    solute.results["dipole_iter_count_vacuum"] = 0 

    induced_dipole_residual = 1.
    dipole_iter_count_vacuum = 1

    bempp.api.log("PBJ: Starting self-consistent interations for induced dipole in vacuum state")

    while induced_dipole_residual > solute.induced_dipole_iter_tol:

        p12scale_temp = solute.p12scale
        p13scale_temp = solute.p13scale

        solute.p12scale = u12scale
        solute.p13scale = u13scale  # scaling for induced dipole calculation

        dphi_Thole = calculate_coulomb_dphi_multipole_Thole(solute, state = "vacuum")

        solute.p12scale = p12scale_temp
        solute.p13scale = p13scale_temp

        dphi_coul = solute.results["d_phi_coulomb_multipole"] + dphi_Thole

        d_induced_prev = d_induced.copy()

        SOR = solute.SOR
        for i in range(N):

            E_total = (dphi_coul[i]/solute.ep_in)*-1
            d_induced[i] = d_induced[i]*(1 - SOR) + np.dot(alphaxx[i], E_total)*SOR

        induced_dipole_residual = np.max(np.sqrt(np.sum(
                            (np.linalg.norm(d_induced_prev-d_induced,axis=1))**2)/len(d_induced)
                        )
                    )

        bempp.api.log("PBJ: Vacuum induced dipole iteration %i -> residual: %s"%(
                    dipole_iter_count_vacuum, induced_dipole_residual
                    )
                )

        dipole_iter_count_vacuum += 1

    solute.results["induced_dipole_vacuum"] = d_induced

def calculate_coulomb_phi_multipole(self):
    """
    Calculate the potential due to the permanent multipoles
    """
    xq = self.x_q
    q = self.q
    d = self.d
    Q = self.Q

    phi = _calculate_coulomb_phi_multipole(xq, q, d, Q)

    return phi 

#@staticmethod
@jit(nopython=True)
def _calculate_coulomb_phi_multipole(xq, q, d, Q):
    """
    Performs calculation of potencial due to permanent multipoles with jit
    """
    N = len(xq)
    eps = 1e-15
    phi = np.zeros(N)

    T2 = np.zeros((N-1,3,3))

    for i in range(N):

        Ri = xq[i] - xq
        Rnorm = np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)

        Ri = np.delete(Ri, (3*i, 3*i+1, 3*i+2)).reshape((N-1,3))
        Rnorm = np.delete(Rnorm, i)
        q_temp = np.delete(q, i)
        d_temp = np.delete(d, (3*i, 3*i+1, 3*i+2)).reshape((N-1,3))
        Q_temp = np.delete(Q, (9*i, 9*i+1, 9*i+2, 9*i+3, 9*i+4, 9*i+5, 9*i+6, 9*i+7, 9*i+8)).reshape((N-1,3,3))

        T0 = 1./Rnorm[:]
        T1 = np.transpose(Ri.transpose()/Rnorm**3)
        T2[:,:,:] = np.ones((N-1,3,3))[:] * Ri.reshape((N-1,1,3)) * \
                    np.transpose(np.ones((N-1,3,3))*Ri.reshape((N-1,1,3)), (0,2,1))/ \
                    Rnorm.reshape((N-1,1,1))**5
        phi[i] = np.sum(q_temp[:]*T0[:]) + np.sum(T1[:]*d_temp[:]) + 0.5*np.sum(np.sum(T2[:]*Q_temp[:],axis=1))

    return phi

def calculate_coulomb_dphi_multipole(self, flag_polar_group=True):
    """
    Calculates the first derivative of the potential due to the permanent multipoles

    flag_polar_group: (bool) consider polar groups in calculation
    """


    xq = self.x_q
    q = self.q
    d = self.d
    Q = self.Q
    alphaxx = self.alpha[:,0,0]
    thole = self.thole
    polar_group = self.polar_group

    dphi = _calculate_coulomb_dphi_multipole(xq, q, d, Q, alphaxx, thole, polar_group, flag_polar_group)

    return dphi


#@staticmethod
@jit(
    nopython=True, parallel=False, error_model="numpy", fastmath=True
)
def _calculate_coulomb_dphi_multipole(xq, q, d, Q, alphaxx, thole, polar_group, flag_polar_group):
    """
    Calculates the first derivative of the potential due to the permanent multipoles
    with numba jit

    flag_polar_group: (bool) consider polar groups in calculation
    """

    N = len(xq)
    T1 = np.zeros((3))
    T2 = np.zeros((3,3))
    eps = 1e-15

    scale3 = 1.0
    scale5 = 1.0
    scale7 = 1.0

    dphi = np.zeros((N,3))

    for i in range(N):

        aux = np.zeros((3))

        Ri = xq[i] - xq
        Rnorm = np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)

        for j in np.where(Rnorm>1e-12)[0]:

            R3 = Rnorm[j]**3
            R5 = Rnorm[j]**5
            R7 = Rnorm[j]**7

            if flag_polar_group==False:

                not_same_polar_group = True

            else:

                gamma = min(thole[i], thole[j])
                damp = (alphaxx[i]*alphaxx[j])**0.16666667
                damp += 1e-12
                damp = -1*gamma * (R3/(damp*damp*damp))
                expdamp = np.exp(damp)

                scale3 = 1 - expdamp
                scale5 = 1 - expdamp*(1-damp)
                scale7 = 1 - expdamp*(1-damp+0.6*damp*damp)

                if polar_group[i]!=polar_group[j]:

                    not_same_polar_group = True

                else:

                    not_same_polar_group = False

            if not_same_polar_group==True:

                for k in range(3):

                    T0 = -Ri[j,k]/R3 * scale3

                    for l in range(3):

                        dkl = (k==l)*1.0

                        T1[l] = dkl/R3 * scale3 - 3*Ri[j,k]*Ri[j,l]/R5 * scale5

                        for m in range(3):

                            dkm = (k==m)*1.0
                            T2[l][m] = (dkm*Ri[j,l]+dkl*Ri[j,m])/R5 * scale5 - 5*Ri[j,l]*Ri[j,m]*Ri[j,k]/R7 * scale7


                    aux[k] += T0*q[j] + np.sum(T1*d[j]) + 0.5*np.sum(np.sum(T2[:,:]*Q[j,:,:], axis = 1), axis = 0)

        dphi[i,:] += aux[:]

    return dphi

def calculate_coulomb_ddphi_multipole(self):

    """
    Calculates the second derivative of the electrostatic potential of the permantent multipoles
    """
    xq = self.x_q
    q = self.q
    d = self.d
    Q = self.Q

    ddphi = _calculate_coulomb_ddphi_multipole(xq, q, d, Q)

    return ddphi

#@staticmethod
@jit(
    nopython=True, parallel=False, error_model="numpy", fastmath=True
)
def _calculate_coulomb_ddphi_multipole(xq, q, d, Q):

    """
    Calculates the second derivative of the electrostatic potential of the permantent multipoles
    with numba jit
    """
    T1 = np.zeros((3))
    T2 = np.zeros((3,3))

    eps = 1e-15

    N = len(xq)

    ddphi = np.zeros((N,3,3))

    for i in range(N):

        aux = np.zeros((3,3))

        Ri = xq[i] - xq
        Rnorm = np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)

        for j in np.where(Rnorm>1e-12)[0]:

            R3 = Rnorm[j]**3
            R5 = Rnorm[j]**5
            R7 = Rnorm[j]**7
            R9 = R3**3

            for k in range(3):

                for l in range(3):

                    dkl = (k==l)*1.0
                    T0 = -dkl/R3 + 3*Ri[j,k]*Ri[j,l]/R5

                    for m in range(3):

                        dkm = (k==m)*1.0
                        dlm = (l==m)*1.0

                        T1[m] = -3*(dkm*Ri[j,l]+dkl*Ri[j,m]+dlm*Ri[j,k])/R5 + 15*Ri[j,l]*Ri[j,m]*Ri[j,k]/R7

                        for n in range(3):

                            dkn = (k==n)*1.0
                            dln = (l==n)*1.0

                            T2[m][n] = 35*Ri[j,k]*Ri[j,l]*Ri[j,m]*Ri[j,n]/R9 - 5*(Ri[j,m]*Ri[j,n]*dkl \
                                                                          + Ri[j,l]*Ri[j,n]*dkm \
                                                                          + Ri[j,m]*Ri[j,l]*dkn \
                                                                          + Ri[j,k]*Ri[j,n]*dlm \
                                                                          + Ri[j,m]*Ri[j,k]*dln)/R7 + (dkm*dln + dlm*dkn)/R5

                    aux[k][l] += T0*q[j] + np.sum(T1[:]*d[j,:]) +  0.5*np.sum(np.sum(T2[:,:]*Q[j,:,:], axis = 1), axis = 0)

        ddphi[i,:,:] += aux[:,:]

    return ddphi

def calculate_coulomb_phi_multipole_Thole(self, state):
    """
    Calculates the potential due to the induced dipoles according to Thole

    state: (string) dissolved or vacuum, to choose which induced dipole to use
    """

    xq = self.x_q
    if state == "dissolved":
        induced_dipole = self.results["induced_dipole"]
    else:
        induced_dipole = self.results["induced_dipole_vacuum"]

    thole = self.thole
    alphaxx = self.alpha[:,0,0]
    connections_12 = self.connections_12
    pointer_connections_12 = self.pointer_connections_12
    connections_13 = self.connections_13
    pointer_connections_13 = self.pointer_connections_13
    p12scale = self.p12scale
    p13scale = self.p13scale

    phi = _calculate_coulomb_phi_multipole_Thole(xq, induced_dipole, thole, alphaxx, connections_12, pointer_connections_12, connections_13, pointer_connections_13, p12scale, p13scale)

    return phi

#@staticmethod
@jit(
    nopython=True, parallel=False, error_model="numpy", fastmath=True
)
def _calculate_coulomb_phi_multipole_Thole(xq, induced_dipole, thole, alphaxx, connections_12, pointer_connections_12, connections_13, pointer_connections_13, p12scale, p13scale):
    """
    Calculates the potential due to the induced dipoles according to Thole
    with numba jit

    """

    eps = 1e-15
    T1 = np.zeros((3))

    N = len(xq)

    phi = np.zeros((N))

    for i in range(N):

        aux = 0.
        start_12 = pointer_connections_12[i]
        stop_12 = pointer_connections_12[i+1]
        start_13 = pointer_connections_13[i]
        stop_13 = pointer_connections_13[i+1]

        Ri = xq[i] - xq

        r = 1./np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)

        for j in np.where(r<1e12)[0]:

            pscale = 1.0

            for ii in range(start_12, stop_12):

                if connections_12[ii]==j:

                    pscale = p12scale

            for ii in range(start_13, stop_13):

                if connections_13[ii]==j:

                    pscale = p13scale

            r3 = r[j]**3

            gamma = min(thole[i], thole[j])
            damp = (alphaxx[i]*alphaxx[j])**0.16666667
            damp += 1e-12
            damp = -gamma * (1/(r3*damp**3))
            expdamp = np.exp(damp)

            scale3 = 1 - expdamp

            for k in range(3):

                T1[k] = Ri[j,k]*r3*scale3*pscale

            aux += np.sum(T1[:]*induced_dipole[j,:])

        phi[i] += aux

    return phi

def calculate_coulomb_dphi_multipole_Thole(self, state):

    """
    Calculates the derivative of the potential due to the induced dipoles according to Thole

    state: (string) dissolved or vacuum, to choose which induced dipole to use
    """

    xq = self.x_q
    if state == "dissolved":
        induced_dipole = self.results["induced_dipole"]
    elif state == "vacuum":
        induced_dipole = self.results["induced_dipole_vacuum"]
    else:
        print("Cannot understand state")

    thole = self.thole
    connections_12 = self.connections_12
    pointer_connections_12 = self.pointer_connections_12
    connections_13 = self.connections_13
    pointer_connections_13 = self.pointer_connections_13
    p12scale = self.p12scale
    p13scale = self.p13scale
    alphaxx = self.alpha[:,0,0]

    dphi = _calculate_coulomb_dphi_multipole_Thole(xq, induced_dipole, thole, alphaxx, connections_12, pointer_connections_12, connections_13, pointer_connections_13, p12scale, p13scale)

    return dphi

#@staticmethod
@jit(
    nopython=True, parallel=False, error_model="numpy", fastmath=True
)
def _calculate_coulomb_dphi_multipole_Thole(xq, induced_dipole, thole, alphaxx, connections_12, pointer_connections_12, connections_13, pointer_connections_13, p12scale, p13scale):

    """
    Calculates the derivative of the potential due to the induced dipoles according to Thole
    with numba jit
    """

    eps = 1e-15
    T1 = np.zeros((3))

    N = len(xq)

    dphi = np.zeros((N,3))

    for i in range(N):

        aux = np.zeros((3))

        start_12 = pointer_connections_12[i]
        stop_12 = pointer_connections_12[i+1]
        start_13 = pointer_connections_13[i]
        stop_13 = pointer_connections_13[i+1]

        Ri = xq[i] - xq
        r = 1./np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)

        for j in np.where(r<1e12)[0]:

            pscale = 1.0

            for ii in range(start_12, stop_12):

                if connections_12[ii]==j:

                    pscale = p12scale

            for ii in range(start_13, stop_13):

                if connections_13[ii]==j:

                    pscale = p13scale

            r3 = r[j]**3
            r5 = r[j]**5

            gamma = min(thole[i], thole[j])
            damp = (alphaxx[i]*alphaxx[j])**0.16666667
            damp += 1e-12
            damp = -gamma * (1/(r3*damp**3))
            expdamp = np.exp(damp)

            scale3 = 1 - expdamp
            scale5 = 1 - expdamp*(1 - damp)

            for k in range(3):

                for l in range(3):

                    dkl = (k==l)*1.0
                    T1[l] = scale3*dkl*r3*pscale - scale5*3*Ri[j,k]*Ri[j,l]*r5*pscale

                aux[k] += np.sum(T1[:] * induced_dipole[j,:])

        dphi[i,:] += aux[:]

    return dphi

def calculate_coulomb_ddphi_multipole_Thole(self, state):
    """
    Calculates the second derivative of the potential due to the induced dipoles according to Thole

    state: (string) dissolved or vacuum, to choose which induced dipole to use
    """

    xq = self.x_q

    if state == "dissolved":
        induced_dipole = self.results["induced_dipole"]
    else:
        induced_dipole = self.results["induced_dipole_vacuum"]

    thole = self.thole
    connections_12 = self.connections_12
    pointer_connections_12 = self.pointer_connections_12
    connections_13 = self.connections_13
    pointer_connections_13 = self.pointer_connections_13
    p12scale = self.p12scale
    p13scale = self.p13scale
    alphaxx = self.alpha[:,0,0]

    ddphi = _calculate_coulomb_ddphi_multipole_Thole(xq, induced_dipole, thole, alphaxx, connections_12, pointer_connections_12, connections_13, pointer_connections_13, p12scale, p13scale)

    return ddphi

#@staticmethod
@jit(
    nopython=True, parallel=False, error_model="numpy", fastmath=True
)
def _calculate_coulomb_ddphi_multipole_Thole(xq, induced_dipole, thole, alphaxx, connections_12, pointer_connections_12, connections_13, pointer_connections_13, p12scale, p13scale):

    """
    Calculates the second derivative of the potential due to the induced dipoles according to Thole
    with numba jit
    """

    eps = 1e-15
    T1 = np.zeros((3))

    N = len(xq)

    ddphi = np.zeros((N,3,3))

    for i in range(N):

        aux = np.zeros((3,3))

        start_12 = pointer_connections_12[i]
        stop_12 = pointer_connections_12[i+1]
        start_13 = pointer_connections_13[i]
        stop_13 = pointer_connections_13[i+1]

        Ri = xq[i] - xq
        r = 1./np.sqrt(np.sum((Ri*Ri), axis = 1) + eps*eps)

        for j in np.where(r<1e12)[0]:

            pscale = 1.0

            for ii in range(start_12, stop_12):

                if connections_12[ii]==j:

                    pscale = p12scale

            for ii in range(start_13, stop_13):

                if connections_13[ii]==j:

                    pscale = p13scale

            r3 = r[j]**3
            r5 = r[j]**5
            r7 = r[j]**7

            gamma = min(thole[i], thole[j])
            damp = (alphaxx[i]*alphaxx[j])**0.16666667
            damp += 1e-12
            damp = -gamma * (1/(r3*damp**3))
            expdamp = np.exp(damp)

            scale5 = 1 - expdamp*(1 - damp)
            scale7 = 1 - expdamp*(1 - damp + 0.6*damp**2)

            for k in range(3):

                for l in range(3):

                    dkl = (k==l)*1.0

                    for m in range(3):

                        dkm = (k==m)*1.0
                        dlm = (l==m)*1.0

                        T1[m] = -3*(dkm*Ri[j,l] + dkl*Ri[j,m] + dlm*Ri[j,k])*r5*scale5*pscale \
                        + 15*Ri[j,l]*Ri[j,m]*Ri[j,k]*r7*scale7*pscale

                    aux[k][l] += np.sum(T1[:]*induced_dipole[j,:])

        ddphi[i,:,:] += aux[:,:]

    return ddphi