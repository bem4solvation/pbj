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
    force_vals_ef = []
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
        force_vals_ef.append(np.linalg.norm(f_solv_energy_func))

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
    file.write("Binding energy value (kcal/mol): 3.9689 \n \n")

    val_energy = (
        np.log((energy_vals[2] - energy_vals[1]) / (energy_vals[1] - energy_vals[0]))
    ) / np.log(2)
    val_force_mst = (
        np.log(
            (force_vals_mst[2] - force_vals_mst[1])
            / (force_vals_mst[1] - force_vals_mst[0])
        )
    ) / np.log(2)
    val_force_ef = (
        np.log(
            (force_vals_ef[2] - force_vals_ef[1])
            / (force_vals_ef[1] - force_vals_ef[0])
        )
    ) / np.log(2)
    file.write(
        f"p- observed convergence value for binding energy (2,4,8): {abs(val_energy):.4f}\n"
    )
    file.write(
        f"p- observed convergence value for force (Maxwell stresses) (2,4,8): {abs(val_force_mst):.4f}\n"
    )
    file.write(
        f"p- observed convergence value for force (Energy functional) (2,4,8): {abs(val_force_ef):.4f}\n"
    )

    reference_energy = 3.9689
    reference_force = 1.9425
    rel_error_energy = abs(energy_vals[-1] - reference_energy) / abs(reference_energy)
    rel_error_force_mst = abs(force_vals_mst[-1] - reference_force) / abs(
        reference_force
    )
    rel_error_force_ef = abs(force_vals_ef[-1] - reference_force) / abs(reference_force)
    file.write(f"\nRelative error for binding energy: {rel_error_energy:.4e}\n")
    file.write(
        f"Relative error for force (Maxwell stresses): {rel_error_force_mst:.4e}\n"
    )
    file.write(
        f"Relative error for force (Energy functional): {rel_error_force_ef:.4e}\n"
    )

    np.testing.assert_allclose(energy_vals[-1], reference_energy, rtol=5e-3)
    np.testing.assert_allclose(force_vals_mst[-1], reference_force, rtol=5e-2)
    np.testing.assert_allclose(force_vals_ef[-1], reference_force, rtol=5e-2)

    file.close()
