# import sys
# import os
# sys.path.insert(0, os.path.abspath("../.."))
import pbj
from pbj.implicit_solvent.utils.analytical import an_P

# from inspect import getmembers, ismodule, isfunction
import numpy as np
from pbj import PBJ_PATH
import os

# run with: python -m pytest -qq test_single.py -s


def test_single():

    def spheres():
        """Generate a list of standard sphere solute meshes at powers-of-two densities.

        Loads the ion_born pqr file (sphere radius 1, charge 1) file and generates
        four distinct sphere meshes with exponentially increasing mesh densities (2, 4, 8, and 16) using
        the MSMS generator.

        Returns:
            list of pbj.implicit_solvent.Solute: A list containing four initialized Solute
                                                 mesh objects corresponding to the
                                                 specified mesh densities.
        """
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
        """Calculate the analytical reaction potential of a Born Ion.

        Computes the reaction potential at a given distance or array of distances
        from the center of a single-charge spherical solute

        Args:
            r (float or array_like): Radial distance(s) from the center of the ion where
                                     the potential is evaluated.
            solute (pbj.Solute): Solute object containing physical properties:
                                  - ep_in: Permittivity inside the ion (epsilon_1)
                                  - ep_ex: Permittivity of the solvent (epsilon_2)
                                  - kappa: Debye-Hückel screening parameter
                                  - r_q: Array of charge radii (r_q[0] is used)
                                  - q: Array of charges (q[0] is used)

        Returns:
            float or array_like: The reaction potential (phi_reac) value(s) corresponding
                                 to the input distance(s).
        """
        epsilon_1 = solute.ep_in
        epsilon_2 = solute.ep_ex
        kappa = solute.kappa
        R = solute.r_q[0]
        q = solute.q[0]

        f_IN = lambda r: q * (
            -1 / (epsilon_1 * R) + 1 / (epsilon_2 * (1 + kappa * R) * R)
        )
        f_OUT = lambda r: q * (
            np.exp(-kappa * (r - R)) / (epsilon_2 * (1 + kappa * R) * r)
            - 1 / (epsilon_1 * r)
        )
        phi_reac = np.piecewise(r, [r <= R, r > R], [f_IN, f_OUT])
        return phi_reac

    def analytic_Born_Ion_vacuum(r, solute):
        """Calculate the analytical Coulomb potential of a Born Ion in a vacuum/uniform medium.

        Args:
            r (float or array_like): Radial distance(s) from the center of the ion where
                                     the potential is evaluated.
            solute (pbj.Solute): Solute object containing physical properties:
                                  - ep_in: Permittivity inside the ion (epsilon_1)
                                  - q: Array of charges (q[0] is used)

        Returns:
            float or array_like: The vacuum potential value(s) corresponding to the
                                 input distance(s).
        """
        epsilon_1 = solute.ep_in
        q = solute.q[0]
        return q / (epsilon_1 * r)

    spheres = spheres()
    solvation_value = an_P(
        spheres[0].q,  # charge
        spheres[0].x_q,  # position of the charge
        spheres[0].ep_in,  # dielectric constant inside the sphere
        spheres[0].ep_ex,  # dielectric constant outside the sphere
        1,  # radius of the sphere
        spheres[0].kappa,  # reciprocal of Debye length
        1,  # radius of the Stern Layer
        3,  # number of terms desired in the polinomial expansion
    )

    energy_vals = []
    file = open("test_results_single.txt", "w")
    for sphere in spheres:
        simulation = pbj.implicit_solvent.Simulation()
        simulation.add_solute(sphere)
        simulation.calculate_solvation_forces(force_formulation="maxwell_tensor")
        f_solv_maxwell_tensor = sphere.results["f_solv"]
        simulation.calculate_solvation_forces(force_formulation="energy_functional")
        f_solv_energy_func = sphere.results["f_solv"]
        simulation.calculate_solvation_energy()
        energy_val = sphere.results["electrostatic_solvation_energy"]
        energy_vals.append(energy_val)
        file.write(
            f"Mesh density: {sphere.mesh_density}, Solvation energy: {energy_val:.4f} "
            f"Solvation forces (Maxwell tensor): {np.linalg.norm(f_solv_maxwell_tensor):.4f} "
            f"Solvation forces (Energy functional): {np.linalg.norm(f_solv_energy_func):.4f}\n"
        )
    file.write(
        f"Analytical solvation energy (kcal/mol): {float(solvation_value):.4f}\n"
    )
    file.write(f"Analytical solvation forces (kcal/molA): {0.0}\n")

    rel_energy_func = lambda E: (E - solvation_value) / solvation_value
    rel_energy_vals = np.array([rel_energy_func(E) for E in energy_vals])
    p_vals_energy = (1 / np.log(2)) * np.log(rel_energy_vals[1:] / rel_energy_vals[:-1])
    file.write(
        f"p-analytical convergence value for energy (4,8,16): {[f'{abs(val):.4f}' for val in p_vals_energy]}\n"
    )

    qe = 1.60217663e-19
    eps0 = 8.8541878128e-12
    ang_to_m = 1e-10
    ke = 1 / (4 * np.pi * eps0)
    to_V = (ke * qe) / ang_to_m  # to_V ≈ 14.39965 V
    r_test = np.array([0.2, 0.8, 1.2, 1.5, 2.0])  # in Angstroms
    file.write("\n\nPotential values (V) at different points:\n")
    file.write(f"r(Ang) : {r_test} \n")

    vals_solv_p = []
    for sphere in spheres:
        simulation = pbj.implicit_solvent.Simulation()
        simulation.add_solute(sphere)
        vals_solute, _ = simulation.calculate_reaction_potential_solute(
            np.array([[r, 0, 0] for r in r_test])
        )
        vals_solute_coul, _ = simulation.calculate_coulomb_potential_solute(
            np.array([[r, 0, 0] for r in r_test])
        )
        vals_solv, _ = simulation.calculate_potential_solvent(
            np.array([[r, 0, 0] for r in r_test])
        )
        vals_solv_p.append(vals_solv[-3] / 1000)
        solute_str = ", ".join([f"{x / 1000:.4f}" for x in vals_solute])
        solute_coul_str = ", ".join([f"{x / 1000:.4f}" for x in vals_solute_coul])
        solv_str = ", ".join([f"{x / 1000:.4f}" for x in vals_solv])
        file.write(
            f"Mesh density: {sphere.mesh_density}, "
            f"Reac potential Solute: [{solute_str}] "
            f"Coulomb potential Solute: [{solute_coul_str}] "
            f"Total potential Solvent: [{solv_str}]\n"
        )
    file.write("\n\nAnalytical potential values (V) at different points\n")

    analytical_potential = []
    for r in r_test:
        phi_r = to_V * analytic_Born_Ion_reac(r, spheres[0])
        phi_v = to_V * analytic_Born_Ion_vacuum(r, spheres[0])
        phi_total = phi_v + phi_r
        analytical_potential.append(phi_total)
        file.write(
            f"r: {r:.2f} Å, phi_reac: {phi_r:.4f} V, phi_vacuum: {phi_v:.4f} V, phi_total: {phi_total:.4f} V \n"
        )

    rel_potential = (
        lambda phi: (phi - analytical_potential[-3]) / analytical_potential[-3]
    )
    rel_potential_vals = np.array([rel_potential(phi) for phi in vals_solv_p])
    p_vals_potential = (1 / np.log(2)) * np.log(
        rel_potential_vals[1:] / rel_potential_vals[:-1]
    )
    file.write(
        f"p-analytical convergence value for potential at 1.2 A (4,8,16): {[f'{abs(val):.4f}' for val in p_vals_potential]}\n"
    )

    # Calculate and print relative errors
    energy_val_extrapolated = energy_vals[-1] + (energy_vals[-1] - energy_vals[-2]) / (
        (2) ** 1 - 1
    )
    rel_error_energy = abs(energy_val_extrapolated - solvation_value) / abs(
        solvation_value
    )
    rel_error_potential = abs(vals_solv_p[-1] - analytical_potential[-3]) / abs(
        analytical_potential[-3]
    )

    print(
        f"\nRelative error for solvation energy (extrapolated r=1): {rel_error_energy:.4e}"
    )
    print(f"Relative error for potential at 1.2 A: {rel_error_potential:.4e}")

    file.write(
        f"\nExtrapolated solvation energy at r=1: {energy_val_extrapolated:.4f} kcal/mol\n"
    )
    file.write(
        f"\nRelative error for solvation energy (extrapolated r=1): {rel_error_energy:.4e}\n"
    )
    file.write(f"Relative error for potential at 1.2 A: {rel_error_potential:.4e}\n")

    np.testing.assert_allclose(energy_val_extrapolated, solvation_value, rtol=1e-2)
    np.testing.assert_allclose(vals_solv_p[-1], analytical_potential[-3], rtol=1e-2)

    file.close()
