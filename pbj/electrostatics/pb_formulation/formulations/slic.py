import numpy as np
import bempp.api
import os
import shutil
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
import pbj
from .common import calculate_potential_stern

def verify_parameters(self):
    return True

def calculate_potential(self, rerun_all):
    dirichl_space_diel = self.dirichl_space
    neumann_space_diel = self.neumann_space
    operator_assembler = self.operator_assembler

    q = self.q
    x_q = self.x_q
    ep_in = self.ep_in
    ep_ex = self.ep_ex
    ep_stern = getattr(self, 'ep_stern', self.ep_ex)
    self.ep_stern = ep_stern

    dlp_adj_in = bempp.api.operators.boundary.laplace.adjoint_double_layer(
    dirichl_space_diel, dirichl_space_diel, neumann_space_diel, assembler=operator_assembler
    )
    identity_diel = sparse.identity(dirichl_space_diel, dirichl_space_diel, dirichl_space_diel, assembler=operator_assembler
    )
    slp_in_diel = laplace.single_layer(
        neumann_space_diel, dirichl_space_diel, dirichl_space_diel, assembler=operator_assembler
    )
    dlp_in_diel = laplace.double_layer(
        dirichl_space_diel, dirichl_space_diel, dirichl_space_diel, assembler=operator_assembler
    )

    getattr(self, 'stern_object', create_stern_mesh(self))

    alpha = self.slic_alpha
    beta = self.slic_beta 
    gamma = self.slic_gamma
    
    max_iterations = self.slic_max_iterations
    tolerance =   self.slic_tolerance

    it = 0
    phi_L2error = 1.

    d2 = 1
    qtotal = np.sum(self.q)
    d1 = - qtotal/self.ep_stern 

    sigma = bempp.api.GridFunction(dirichl_space_diel, coefficients = np.zeros(
    dirichl_space_diel.global_dof_count))

    @bempp.api.real_callable
    def en_function(x, n, domain_index, result):
        nrm = np.sqrt(
                (x[0] - x_q[:, 0]) ** 2
                + (x[1] - x_q[:, 1]) ** 2
                + (x[2] - x_q[:, 2]) ** 2
            )
        result[:] = np.sum(np.dot(n,(np.transpose(x-x_q)/
                                 (4 * np.pi * nrm**3))*
                              (q)))

    electricfield = bempp.api.GridFunction(dirichl_space_diel, fun=en_function)

    while it < max_iterations and phi_L2error > tolerance:

        if it == 0:
            self.e_hat_diel = self.ep_in / self.ep_stern 
            self.e_hat_stern = self.ep_stern / self.ep_ex

        else: 
            Ksigma = dlp_adj_in * sigma
            en = electricfield - Ksigma
            mu = - alpha * np.tanh(- gamma)
            h = alpha * np.tanh(beta * en.coefficients - gamma) + mu
            f = ep_in/(ep_stern - ep_in) - h
            f_div = f/(1+f)
            f_fun = bempp.api.GridFunction(neumann_space_diel, coefficients = f_div)
            self.e_hat_diel = bempp.api.assembly.boundary_operator.MultiplicationOperator(f_fun, 
                                neumann_space_diel, neumann_space_diel, neumann_space_diel)
            d2 = self.results['d_phi_stern'].integrate()[0] 
            self.e_hat_stern = d1/d2
            phi_old = self.results['phi'].copy()

        calculate_potential_stern(self, rerun_all = True)

        rhs_sigma = slp_in_diel * self.results['d_phi'] - (- 0.5 * identity_diel + dlp_in_diel) * self.results['phi']
        sigma,_ = bempp.api.linalg.gmres(slp_in_diel, rhs_sigma, tol= self.gmres_tolerance, maxiter = self.gmres_max_iterations)  

        phi_L2error = np.sqrt(np.sum((phi_old-self.results['phi'])**2)/np.sum(self.results['phi']**2))

            
            

        


        
        

        





    
    

    


    self.e_hat_diel = self.ep_in / self.ep_stern 
    self.e_hat_stern = self.ep_stern / self.ep_ex

    calculate_potential_stern(self, rerun_all)