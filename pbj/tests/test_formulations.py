# import sys
# import os
# sys.path.insert(0, os.path.abspath("../.."))
import pbj
import pbj.implicit_solvent.pb_formulation.formulations as pb_formulations
from pbj.implicit_solvent.utils.analytical import an_P
from inspect import getmembers, ismodule, isfunction
import numpy as np
from pbj import PBJ_PATH
import os

# run with: python -m pytest -qq test_formulations.py -s


def test_formulations():
    def richardson_extrapolation(f1, f2, f3, r):
        """Perform Richardson extrapolation to estimate the order of accuracy
        and the continuum-truncated exact solution.

        Args:
            f1 (float or array_like): Solution on the finest grid.
            f2 (float or array_like): Solution on the medium grid.
            f3 (float or array_like): Solution on the coarsest grid.
            r (float): Grid refinement ratio (e.g., r = h_coarse / h_fine).

        Returns:
            p (float or array_like): Observed order of accuracy.
            f (float or array_like): Extrapolated value (zero-grid-spacing estimate).
        """
        p = np.log((f3 - f2) / (f2 - f1)) / np.log(r)
        f = f1 + (f1 - f2) / (r**p - 1)
        return p, f

    def spheres():
        """Generate a list of sphere solute meshes at different grid densities.

        Loads a test PQR file and generates three distinct sphere meshes with
        varying mesh densities using the MSMS generator.

        Returns:
            list of pbj.Solute: A list containing three initialized Solute mesh objects
                                corresponding to mesh densities of 0.85, 1.7, and 3.4.
        """
        spheres = []
        print("Creating sphere meshes")
        pqrpath = os.path.join(PBJ_PATH, "tests", "test.pqr")
        for mesh_dens in [0.85, 1.7, 3.4]:
            sphere = pbj.Solute(pqrpath, mesh_density=mesh_dens, mesh_generator="msms")
            sphere.x_q[0][0] = 0.1
            spheres.append(sphere)
        return spheres

    def histidines():
        """Generate a list of histidine solute meshes at different grid scales.

        Loads a test histidine PQR file and generates three distinct molecular
        meshes with varying grid scales using the NanoShaper mesh generator.

        Returns:
            list of pbj.Solute: A list containing three initialized Solute mesh objects
                                corresponding to grid scales of 1.4, 1.82, and 2.366.
        """
        histidines = []
        print("Creating histidine meshes")
        pqrpath = os.path.join(PBJ_PATH, "tests", "his", "his.pqr")
        for mesh_dens in [1.4, 1.82, 2.366]:
            histidine = pbj.Solute(
                pqrpath, nanoshaper_grid_scale=mesh_dens, mesh_generator="nanoshaper"
            )
            histidines.append(histidine)
        return histidines

    def values():
        """Dynamically map available Poisson-Boltzmann formulations and their preconditioners.

        Inspects the `pb_formulations` module to identify valid formulation submodules
        (excluding the 'common' module). For each formulation, it scans for available
        preconditioner functions and initializes a nested dictionary structure.

        Returns:
            dict: A nested dictionary where keys are formulation names. Each formulation
                  dictionary contains a `"no_precond"` key and keys for each discovered
                  preconditioner (with the "_preconditioner" suffix stripped), all
                  initialized to empty `np.array([])` objects.
        """
        values = {}
        available = getmembers(pb_formulations, ismodule)
        for element in available:
            if element[0] == "common":
                available.remove(element)
        for formulation_name, object_address in available:
            formulation = getattr(pb_formulations, formulation_name, None)
            values[formulation_name] = {}
            values[formulation_name]["no_precond"] = np.array([])
            for precond_name, object_address in getmembers(formulation, isfunction):
                if precond_name.endswith("preconditioner"):
                    precond_name_removed = precond_name[:-15]
                    values[formulation_name][precond_name_removed] = np.array([])
        return values

    spheres = spheres()
    histidines = histidines()
    values = values()
    file = open("test_results.txt", "w")
    solvation_value = an_P(
        spheres[0].q,
        spheres[0].x_q,
        spheres[0].ep_in,
        spheres[0].ep_ex,
        5,
        spheres[0].kappa,
        5,
        3,
    )
    solvation_value_stern = an_P(
        spheres[0].q,
        spheres[0].x_q,
        spheres[0].ep_in,
        spheres[0].ep_ex,
        5,
        spheres[0].kappa,
        7,
        3,
    )
    his_1 = -25.812683243090387
    his_2 = -24.48630195977033
    his_3 = -23.875642908991132
    _, solvation_value_his = richardson_extrapolation(his_3, his_2, his_1, 1.3)
    tol = 0.5  # cambiar a reltol 0.05% ideal, 0.01% probar
    file.write(
        "The expected value for the solvation of the sphere is: {}, with a tolerance of {}.\n\n".format(
            solvation_value, tol
        )
    )
    file.write(
        "The expected value for the solvation of an histidine aminoacid (used to verificate the SLIC and SLIC_PROP formulations) is: {}, with a tolerance of {}.\n\n".format(
            solvation_value_his, tol
        )
    )

    for sphere in spheres:
        formulations = list(values.keys())
        formulations.remove("slic")
        formulations.remove("slic_prop")
        for formulation in formulations:
            print(
                "Computing for {} with {}".format(formulation, sphere.sas_mesh_density)
            )
            for preconditioner in values[formulation].keys():
                simulation = pbj.implicit_solvent.Simulation()
                simulation.pb_formulation = formulation
                simulation.add_solute(sphere)
                if preconditioner == "no_precond":
                    simulation.solutes[0].pb_formulation_preconditioning = False
                    simulation.calculate_solvation_energy()
                    values[formulation]["no_precond"] = np.append(
                        values[formulation]["no_precond"],
                        (
                            simulation.solutes[0].results[
                                "electrostatic_solvation_energy"
                            ],
                            sphere.sas_mesh_density,
                        ),
                    )
                else:
                    simulation.solutes[0].pb_formulation_preconditioning = True
                    simulation.solutes[0].pb_formulation_preconditioning_type = (
                        preconditioner
                    )
                    simulation.calculate_solvation_energy()
                    values[formulation][preconditioner] = np.append(
                        values[formulation][preconditioner],
                        (
                            simulation.solutes[0].results[
                                "electrostatic_solvation_energy"
                            ],
                            sphere.sas_mesh_density,
                        ),
                    )

    for his in histidines:
        formulations = ["slic", "slic_prop"]
        for formulation in formulations:
            print(
                "Computing for histidine, {} with {}".format(
                    formulation, his.nanoshaper_grid_scale
                )
            )
            for preconditioner in values[formulation].keys():
                simulation = pbj.implicit_solvent.Simulation()
                simulation.pb_formulation = formulation
                simulation.add_solute(his)
                if preconditioner == "no_precond":
                    simulation.solutes[0].pb_formulation_preconditioning = False
                    simulation.calculate_solvation_energy()
                    values[formulation]["no_precond"] = np.append(
                        values[formulation]["no_precond"],
                        (
                            simulation.solutes[0].results[
                                "electrostatic_solvation_energy"
                            ],
                            his.nanoshaper_grid_scale,
                        ),
                    )
                else:
                    simulation.solutes[0].pb_formulation_preconditioning = True
                    simulation.solutes[0].pb_formulation_preconditioning_type = (
                        preconditioner
                    )
                    simulation.calculate_solvation_energy()
                    values[formulation][preconditioner] = np.append(
                        values[formulation][preconditioner],
                        (
                            simulation.solutes[0].results[
                                "electrostatic_solvation_energy"
                            ],
                            his.nanoshaper_grid_scale,
                        ),
                    )

    solvation_energy_values = np.array([])
    solvation_energy_expected_values = np.array([])
    solvation_energy_values_formulation_and_precond = np.array([])
    p_array = np.array([])
    file.write(
        "Extrapolated values and p parameter for each formulation and preconditioner combination:\n"
    )

    for formulation in values.keys():
        for preconditioner in values[formulation].keys():
            val_array = values[formulation][preconditioner]
            if formulation in ["slic", "slic_prop"]:
                p, val = richardson_extrapolation(
                    val_array[4], val_array[2], val_array[0], 1.3
                )
            else:
                p, val = richardson_extrapolation(
                    val_array[4], val_array[2], val_array[0], 2
                )
            solvation_energy_values = np.append(solvation_energy_values, val)
            file.write(
                "{} with {}: {}, {}\n".format(formulation, preconditioner, val, p)
            )
            p_array = np.append(p_array, p)
            solvation_energy_values_formulation_and_precond = np.append(
                solvation_energy_values_formulation_and_precond,
                formulation + "_" + preconditioner,
            )
            if formulation in ["direct_stern"]:
                solvation_energy_expected_values = np.append(
                    solvation_energy_expected_values, solvation_value_stern
                )
            elif formulation in ["slic", "slic_prop"]:
                solvation_energy_expected_values = np.append(
                    solvation_energy_expected_values, solvation_value_his
                )
            else:
                solvation_energy_expected_values = np.append(
                    solvation_energy_expected_values, solvation_value
                )

    indexes = list(
        zip(
            *np.where(
                ~np.isclose(
                    solvation_energy_values,
                    solvation_energy_expected_values,
                    atol=tol,
                    rtol=0,
                )
            )
        )
    )
    if len(indexes) > 0:
        file.write(
            "\nThe following combination(s) of formulation and preconditioner did not match the expected value:\n"
        )
        for i in indexes:
            file.write(solvation_energy_values_formulation_and_precond[i] + "\n")
    else:
        file.write("\nAll values match the expected value.\n")
    np.testing.assert_allclose(
        solvation_energy_values, solvation_energy_expected_values, atol=tol, rtol=0
    )
    file.close()
