import sys
import os

# sys.path.insert(0, os.path.abspath("../.."))
import pbj
from pbj.implicit_solvent import simulation
from pbj.implicit_solvent import simulation
import pbj.implicit_solvent.pb_formulation.formulations as pb_formulations
from pbj.implicit_solvent.utils.analytical import an_P
from inspect import getmembers, ismodule, isfunction
import numpy as np
from pbj import PBJ_PATH
import os

# run with: python -m pytest -qq test_single.py -s


def test_single():

    def spheres():
        spheres = []
        print("Creating sphere meshes")
        pqrpath = os.path.join(PBJ_PATH, "tests", "spheres", "test_sphere_born.pqr")
        for mesh_dens in [2, 4, 8, 16]:
            sphere = pbj.Solute(pqrpath, mesh_density=mesh_dens, mesh_generator="msms")
            sphere.ep_in = 1.0
            sphere.x_q[0][0] = 0.00001
            spheres.append(sphere)
        return spheres
    
    def analytic_Born_Ion_reac(r, solute):
        epsilon_1 = solute.ep_in
        epsilon_2 = solute.ep_ex
        kappa = solute.kappa
        R = solute.r_q[0]
        q = solute.q[0]

        f_IN = lambda r: q * ( - 1/(epsilon_1*R) + 1/(epsilon_2*(1+kappa*R)*R) )
        f_OUT = lambda r: q * (np.exp(-kappa*(r-R))/(epsilon_2*(1+kappa*R)*r) - 1/(epsilon_1*r))
        phi_reac = np.piecewise(r, [r<=R, r>R], [f_IN, f_OUT])
        return phi_reac

    def analytic_Born_Ion_vacuum(r, solute):
        epsilon_1 = solute.ep_in
        q = solute.q[0]
        return q / (epsilon_1*r)
    

    solvation_value = an_P(
        spheres[0].q, #charge
        spheres[0].x_q, #position of the charge
        spheres[0].ep_in, #dielectric constant inside the sphere
        spheres[0].ep_ex, #dielectric constant outside the sphere
        1, #radius of the sphere
        spheres[0].kappa, #reciprocal of Debye length
        1, #radius of the Stern Layer
        3, #number of terms desired in the polinomial expansion
    )
    
    spheres = spheres()
    file = open("test_results_single.txt", "w")

    for sphere in spheres:
        simulation = pbj.implicit_solvent.Simulation()
        simulation.add_solute(sphere)
        simulation.calculate_solvation_forces(force_formulation='maxwell_tensor')
        f_solv_maxwell_tensor = sphere.results['f_solv']
        simulation.calculate_solvation_forces(force_formulation='energy_functional')
        f_solv_energy_func = sphere.results['f_solv']
        simulation.calculate_solvation_energy()
        energy_val = sphere.results["electrostatic_solvation_energy"]
        file.write(f"Mesh density: {sphere.mesh_density}, Solvation energy: {energy_val:.4f} " \
                f"Solvation forces (Maxwell tensor): {np.linalg.norm(f_solv_maxwell_tensor):.4f} " \
                f"Solvation forces (Energy functional): {np.linalg.norm(f_solv_energy_func):.4f}\n")
    file.write(f"Analytical solvation energy (kcal/mol): {float(solvation_value):.4f}\n")
    file.write(f"Analytical solvation forces (kcal/molA): {0.0}\n")


    qe = 1.60217663e-19
    eps0 = 8.8541878128e-12
    ang_to_m = 1e-10
    ke = 1 / (4 * np.pi * eps0)
    to_V = (ke * qe) / ang_to_m  # to_V ≈ 14.39965 V 
    r_test = np.array([0.2, 0.8, 1.2, 1.5, 2.0]) # in Angstroms
    file.write("\n\nPotential values (V) at different points:\n")
    file.write(f"r(Ang) : {r_test} \n")

    for sphere in spheres:
        simulation = pbj.implicit_solvent.Simulation()
        simulation.add_solute(sphere)
        vals_solute, _=simulation.calculate_reaction_potential_solute([[r,0,0] for r in r_test])
        vals_solv, _ = simulation.calculate_potential_solvent([[r,0,0] for r in r_test])
        solute_str = ", ".join([f"{x/1000:.4f}" for x in vals_solute])
        solv_str = ", ".join([f"{x/1000:.4f}" for x in vals_solv])
        file.write(
            f"Mesh density: {sphere.mesh_density}, "
            f"Reac potential Solute: [{solute_str}] "
            f"Total potential Solvent: [{solv_str}]\n"
        )
    file.write("\n\nAnalytical potential values (V) at different points:\n")
    for r in r_test:
        phi_r = to_V * analytic_Born_Ion_reac(r, spheres[0])
        phi_v = to_V * analytic_Born_Ion_vacuum(r, spheres[0])
        phi_total = phi_v + phi_r
        file.write(f"r: {r:.2f} Å, phi_reac: {phi_r:.4f} V, phi_vacuum: {phi_v:.4f} V, phi_total: {phi_total:.4f} V \n")    

    file.close()
