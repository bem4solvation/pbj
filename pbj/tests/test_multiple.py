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
        """Generate a list of standard sphere solute meshes at powers-of-two densities.

        Loads the `test_sphere1.pqr` (sphere radius 2, charge 1) file and generates four
        distinct sphere meshes with exponentially increasing mesh densities (2, 4, 8, and 16)
        using the MSMS generator.

        Returns:
            list of pbj.implicit_solvent.Solute: A list containing four initialized Solute
                                                 mesh objects corresponding to the
                                                 specified mesh densities.
        """
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
        """Generate a list of standard sphere solute meshes at powers-of-two densities.

        Loads the `test_sphere2.pqr` (sphere radius 2, charge 1, distance +3 A) file and generates
        four distinct sphere meshes with exponentially increasing mesh densities (2, 4, 8, and 16) using
        the MSMS generator.

        Returns:
            list of pbj.implicit_solvent.Solute: A list containing four initialized Solute
                                                 mesh objects corresponding to the
                                                 specified mesh densities.
        """
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

    energy_vals = []
    force_vals_mst = []
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
        force_vals_mst.append(np.linalg.norm(f_solv_maxwell_tensor))
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
        energy_vals.append(energy_val)
        file.write(
            f"Mesh density: {spheres1[j].mesh_density}, Binding energy energy: {energy_val:.4f} "
            f"Solvation forces (Maxwell tensor): {np.linalg.norm(f_solv_maxwell_tensor):.4f} "
            f"Solvation forces (Energy functional): {np.linalg.norm(f_solv_energy_func):.4f} \n"
        )

    file.write("\n\nReference values (Paper JCTC 2023):\n")
    file.write("Force value (by virtual work) (kcal/molA): 1.9425 \n")
    file.write("Binding energy value (kcal/mol): 3.9689 \n")

    rel_energy_func = lambda E: (E - 3.9689) / 3.9689
    rel_force_func = lambda E: (E - 1.9425) / 1.9425
    rel_energy_vals = np.array([rel_energy_func(E) for E in energy_vals])
    rel_force_vals = np.array([rel_force_func(E) for E in force_vals_mst])
    p_vals_energy = (1 / np.log(2)) * np.log(rel_energy_vals[1:] / rel_energy_vals[:-1])
    p_vals_force = (1 / np.log(2)) * np.log(rel_force_vals[1:] / rel_force_vals[:-1])
    file.write(
        f"p- observed convergence value for energy (4,8,16): {[f'{abs(val):.4f}' for val in p_vals_energy]}\n"
    )
    file.write(
        f"p- observed convergence value for force (4,8,16): {[f'{abs(val):.4f}' for val in p_vals_force]}\n"
    )

    file.close()
