# import sys
# import os
# sys.path.insert(0, os.path.abspath("../.."))
import pbj

# from inspect import getmembers, ismodule, isfunction
import numpy as np
from pbj import PBJ_PATH
import os

# run with: python -m pytest -qq test_multiple.py -s


def test_multiple():

    def spheres1():
        spheres = []
        print("Creating sphere meshes")
        pqrpath = os.path.join(PBJ_PATH, "tests", "spheres", "test_sphere1.pqr")
        for mesh_dens in [2, 4, 8, 16]:
            sphere = pbj.implicit_solvent.Solute(
                pqrpath, mesh_density=mesh_dens, mesh_generator="msms"
            )
            spheres.append(sphere)
        return spheres

    def spheres2():
        spheres = []
        print("Creating sphere meshes")
        pqrpath = os.path.join(PBJ_PATH, "tests", "spheres", "test_sphere2.pqr")
        for mesh_dens in [2, 4, 8, 16]:
            sphere = pbj.implicit_solvent.Solute(
                pqrpath, mesh_density=mesh_dens, mesh_generator="msms"
            )
            spheres.append(sphere)
        return spheres

    spheres_sing = spheres1()
    spheres1 = spheres1()
    spheres2 = spheres2()
    file = open("test_results_multiple.txt", "w")
    for j in range(len(spheres1)):
        print(
            f"Calculating solvation energy/forces for mesh density {spheres1[j].mesh_density}"
        )
        simulation = pbj.implicit_solvent.Simulation()
        simulation.add_solute(spheres_sing[j])
        simulation.calculate_solvation_energy()
        energy_val_single = spheres_sing[j].results["electrostatic_solvation_energy"]

        simulation_mult = pbj.implicit_solvent.Simulation()
        simulation_mult.add_solute(spheres1[j])
        simulation_mult.add_solute(spheres2[j])
        simulation_mult.calculate_solvation_forces(force_formulation="maxwell_tensor")
        f_solv_maxwell_tensor = spheres1[j].results["f_solv"]
        simulation_mult.calculate_solvation_forces(
            force_formulation="energy_functional"
        )
        f_solv_energy_func = spheres1[j].results["f_solv"]
        simulation_mult.calculate_solvation_energy()
        energy_val_multiple = (
            spheres1[j].results["electrostatic_solvation_energy"]
            + spheres2[j].results["electrostatic_solvation_energy"]
        )

        energy_val = energy_val_multiple - 2 * energy_val_single
        file.write(
            f"Mesh density: {spheres1[j].mesh_density}, Binding energy energy: {energy_val:.4f} "
            f"Solvation forces (Maxwell tensor): {np.linalg.norm(f_solv_maxwell_tensor):.4f} "
            f"Solvation forces (Energy functional): {np.linalg.norm(f_solv_energy_func):.4f} \n"
        )

    file.write("\n\nReference values (Paper JCTC 2023):\n")
    file.write("Force value (by virtual work) (kcal/molA): 1.9425 \n")
    file.write("Binding energy value (kcal/mol): 3.9689 \n")

    file.close()
