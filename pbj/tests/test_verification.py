import pytest
import pbj
import pbj.electrostatics.pb_formulation.formulations as pb_formulations
from pbj.electrostatics.utils.analytical import an_P
from inspect import getmembers, ismodule, isfunction
import numpy as np
from pbj import PBJ_PATH
import os

# run with: python -m pytest -q test_verification.py -s  


def test_verification():
    
    def richardson_extrapolation(f1,f2,f3,r):
        p = np.log((f3-f2)/(f2-f1))/np.log(r)
        f = f1 + (f1-f2)/(r**p-1)
        return p, f


    def spheres():
        spheres = []
        print("Creating meshes")
        pqrpath = os.path.join(PBJ_PATH, "tests", "test.pqr")
        for mesh_dens in [0.425, 0.85, 1.7]:
            sphere = pbj.Solute(pqrpath, mesh_density=mesh_dens, mesh_generator="msms")
            sphere.x_q[0][0] = 0.1
            spheres.append(sphere)
        return spheres


    def values():
        values = {}
        for formulation_name, object_address in getmembers(pb_formulations, ismodule):
            formulation = getattr(pb_formulations, formulation_name, None)
            values[formulation_name] = {}
            values[formulation_name]["no_precond"] = np.array([])
            for precond_name, object_address in getmembers(formulation, isfunction):
                if precond_name.endswith("preconditioner"):
                    precond_name_removed = precond_name[:-15]
                    values[formulation_name][precond_name_removed] = np.array([])
        return values


    spheres = spheres()
    values = values()
    file = open("test_results.txt", "w")
    solvation_value = an_P(spheres[0].q, spheres[0].x_q, spheres[0].ep_in, spheres[0].ep_ex, 5, spheres[0].kappa, 5, 3)
    tol = 0.1
    file.write("The expected value is: {}, with a tolerance of {}.\n\n".format(solvation_value,tol))
    for sphere in spheres:
        for formulation in values.keys():
            print("Computing for {} with {}".format(formulation, sphere.mesh_density))
            sphere.pb_formulation = formulation
            for preconditioner in values[formulation].keys():
                if preconditioner == "no_precond":
                    sphere.pb_formulation_preconditioning = False
                    sphere.calculate_solvation_energy(rerun_all=True)
                    print(values[formulation]["no_precond"])
                    print(sphere.results["solvation_energy"])
                    print(sphere.mesh_density)
                    values[formulation]["no_precond"] = np.append(
                        values[formulation]["no_precond"],
                        (sphere.results["solvation_energy"], sphere.mesh_density),
                    )
                else:  # preconditioner
                    sphere.pb_formulation_preconditioning = True
                    sphere.pb_formulation_preconditioning_type = preconditioner
                    sphere.calculate_solvation_energy(rerun_all=True)
                    print("Current formulation: {}".format(formulation))
                    print("Current sphere: {}".format(sphere.mesh_density))
                    print("Current preconditioner: {}".format(preconditioner))
                    values[formulation][preconditioner] = np.append(
                        values[formulation][preconditioner],
                        (sphere.results["solvation_energy"], sphere.mesh_density),
                    )

    solvation_energy_values = np.array([])
    solvation_energy_expected_values = np.array([])
    solvation_energy_values_formulation_and_precond = np.array([])
    p_array = np.array([])
    file.write("Extrapolated values and p parameter for each formulation and preconditioner combination:\n")
    for formulation in values.keys():
        for preconditioner in values[formulation].keys():
            val_array = values[formulation][preconditioner]
            p,val = richardson_extrapolation(val_array[4], val_array[2], val_array[0],2)  
            solvation_energy_values = np.append(solvation_energy_values, val)
            file.write("{} with {}: {}, {}\n".format(formulation, preconditioner, val, p))
            p_array = np.append(p_array, p)
            solvation_energy_values_formulation_and_precond = np.append(
                solvation_energy_values_formulation_and_precond,
                formulation + "_" + preconditioner,
            )
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

