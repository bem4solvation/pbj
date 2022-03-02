import pbj
import pbj.electrostatics.pb_formulation.formulations as pb_formulations
from pbj.electrostatics.utils.analytical import an_P
from inspect import getmembers, ismodule, isfunction
import numpy as np
from pbj import PBJ_PATH
import os

# run with: python -m pytest -qq test_verification.py -s


def test_verification():
    def richardson_extrapolation(f1, f2, f3, r):
        p = np.log((f3 - f2) / (f2 - f1)) / np.log(r)
        f = f1 + (f1 - f2) / (r ** p - 1)
        return p, f

    def spheres():
        spheres = []
        print("Creating sphere meshes")
        pqrpath = os.path.join(PBJ_PATH, "tests", "test.pqr")
        #for mesh_dens in [0.85, 1.7, 3.4]:
        for mesh_dens in [0.05,0.1,0,2]:
            sphere = pbj.Solute(pqrpath, mesh_density=mesh_dens, mesh_generator="msms")
            sphere.x_q[0][0] = 0.1
            spheres.append(sphere)
        return spheres

    def histidines():
        histidines = []
        print("Creating histidine meshes")
        pqrpath = os.path.join(PBJ_PATH, "tests", 'his', "his.pqr")
        for mesh_dens in [1.4,1.82,2.366]:
            histidine = pbj.Solute(pqrpath, nanoshaper_grid_scale=mesh_dens, mesh_generator='nanoshaper')
            histidines.append(histidine)
        return histidines

    def values():
        values = {}
        available = getmembers(pb_formulations, ismodule)
        for element in available:
            if element[0] == 'common':
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
    his_3  = -23.875642908991132
    solvation_value_his = richardson_extrapolation(his_3, his_2, his_1, 1.3)
    tol = 0.1
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
            print("Computing for {} with {}".format(formulation, sphere.mesh_density))
            sphere.pb_formulation = formulation
            for preconditioner in values[formulation].keys():
                if preconditioner == "no_precond":
                    print("No preconditioner")
                    sphere.pb_formulation_preconditioning = False
                    sphere.calculate_solvation_energy(rerun_all=True)
                    values[formulation]["no_precond"] = np.append(
                        values[formulation]["no_precond"],
                        (sphere.results["solvation_energy"], sphere.mesh_density),
                    )
                else:
                    sphere.pb_formulation_preconditioning = True
                    sphere.pb_formulation_preconditioning_type = preconditioner
                    sphere.calculate_solvation_energy(rerun_all=True)
                    values[formulation][preconditioner] = np.append(
                        values[formulation][preconditioner],
                        (sphere.results["solvation_energy"], sphere.mesh_density),
                    )

    for his in histidines:
        formulations = ['slic', 'slic_prop']
        for formulation in formulations:
            print("Computing for histidine, {} with {}".format(formulation, his.nanoshaper_grid_scale))
            his.pb_formulation = formulation
            for preconditioner in values[formulation].keys():
                if preconditioner == "no_precond":
                    print("No preconditioner")
                    his.pb_formulation_preconditioning = False
                    his.calculate_solvation_energy(rerun_all=True)
                    values[formulation]["no_precond"] = np.append(
                        values[formulation]["no_precond"],
                        (his.results["solvation_energy"], his.nanoshaper_grid_scale),
                    )
                else:
                    his.pb_formulation_preconditioning = True
                    his.pb_formulation_preconditioning_type = preconditioner
                    his.calculate_solvation_energy(rerun_all=True)
                    values[formulation][preconditioner] = np.append(
                        values[formulation][preconditioner],
                        (his.results["solvation_energy"], his.nanoshaper_grid_scale),
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
            if formulation in ['slic', 'slic_prop']:
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
            if formulation in ['direct_stern']:
                solvation_energy_expected_values = np.append(
                    solvation_energy_expected_values, solvation_value_stern
                )
            elif formulation in ['slic', 'slic_prop']:
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
