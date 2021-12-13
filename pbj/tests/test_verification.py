import pytest
import pbj
import pbj.electrostatics.pb_formulation.formulations as pb_formulations
from inspect import getmembers, ismodule, isfunction
import numpy as np
from pbj import PBJ_PATH
import os


@pytest.fixture
def spheres():
    spheres = []
    print("Creating meshes")
    pqrpath = os.path.join(PBJ_PATH, "tests", "test.pqr")
    for mesh_dens in [0.1, 0.2, 0.3]:
        sphere = pbj.Solute(pqrpath, mesh_density=mesh_dens, mesh_generator="msms")
        sphere.x_q[0][0] = 0.1
        spheres.append(sphere)
    return spheres


@pytest.fixture
def values():
    values = {}
    for formulation_name, object_address in getmembers(pb_formulations, ismodule):
        formulation = getattr(pb_formulations, formulation_name, None)
        values[formulation_name] = {}
        values[formulation_name]["no_precond"] = np.array([])
        for precond_name, object_address in getmembers(formulation, isfunction):
            if precond_name.endswith("preconditioner"):
                precond_name_removed = precond_name[:-15]
                if not precond_name_removed.startswith("calderon"):
                    values[formulation_name][precond_name_removed] = np.array([])
    return values


def test_verification(spheres, values):

    solvation_value = -8.46
    tol = 0.01

    # No preconditioner
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
    for formulation in values.keys():
        for preconditioner in values[formulation].keys():
            val_array = values[formulation][preconditioner]
            val = np.mean([val_array[0], val_array[2], val_array[4]])  # Extrapolation!
            solvation_energy_values = np.append(solvation_energy_values, val)
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
        print(
            "The following combination(s) of formulation and preconditioner did not match the expected value:"
        )
        for i in indexes:
            print(solvation_energy_values_formulation_and_precond[i])
    np.testing.assert_allclose(
        solvation_energy_values, solvation_energy_expected_values, atol=tol, rtol=0
    )


"""
@pytest.mark.parametrize("test_input, expected", zip(solvation_energy_values, solvation_energy_expected_values))
def test_solvation_energy(test_input, expected):
  assert test_input[2] == expected[2]
@pytest.mark.parametrize("test_input, expected", zip(solvation_energy_values, solvation_energy_expected_values))
test_solvation_energy(test_input, expected)


values = {'alpha_beta': {'no_precond': np.array([-8.79048736,  0.1       , -8.38036398,  0.2       , -8.23485811,
        0.3       ]), 'block_diagonal': np.array([-8.79042303,  0.1       , -8.38051119,  0.2       , -8.23502703,
        0.3       ]), 'mass_matrix': np.array([-8.79049514,  0.1       , -8.38037789,  0.2       , -8.23488211,
        0.3       ])}, 'alpha_beta_external_potential': {'no_precond': np.array([-8.79047874,  0.1       , -8.38040494,  0.2       , -8.23491681,
        0.3       ])}, 'alpha_beta_single_blocked': {'no_precond': np.array([-8.79042303,  0.1       , -8.38051119,  0.2       , -8.23502703,
        0.3       ])}, 'direct': {'no_precond': np.array([-8.80811966,  0.1       , -8.38574854,  0.2       , -8.23700007,
        0.3       ]), 'block_diagonal': np.array([-8.80812829,  0.1       , -8.38576258,  0.2       , -8.23705428,
        0.3       ]), 'mass_matrix': np.array([-8.80812878,  0.1       , -8.3857669 ,  0.2       , -8.23705313,
        0.3       ])}, 'direct_external': {'no_precond': np.array([-8.8081285 ,  0.1       , -8.38576476,  0.2       , -8.23705651,
        0.3       ])}, 'direct_external_permuted': {'no_precond': np.array([-8.80812805,  0.1       , -8.38576698,  0.2       , -8.23705799,
        0.3       ])}, 'direct_permuted': {'no_precond': np.array([-8.80812803,  0.1       , -8.38576187,  0.2       , -8.23705379,
        0.3       ]), 'block_diagonal': np.array([-8.80812829,  0.1       , -8.38576258,  0.2       , -8.23705428,
        0.3       ]), 'mass_matrix': np.array([-8.80812878,  0.1       , -8.3857669 ,  0.2       , -8.23705313,
        0.3       ])}, 'first_kind_external': {'no_precond': np.array([-8.80877376,  0.1       , -8.38598447,  0.2       , -8.2371572 ,
        0.3       ]), 'mass_matrix': np.array([-8.80877625,  0.1       , -8.38598313,  0.2       , -8.23715312,
        0.3       ])}, 'first_kind_internal': {'no_precond': np.array([-8.80876792,  0.1       , -8.38597125,  0.2       , -8.23714127,
        0.3       ]), 'mass_matrix': np.array([-8.80876587,  0.1       , -8.3859705 ,  0.2       , -8.23714097,
        0.3       ])}, 'juffer': {'no_precond': np.array([-8.78267069,  0.1       , -8.37756265,  0.2       , -8.23365942,
        0.3       ]), 'block_diagonal': np.array([-8.78260072,  0.1       , -8.37756909,  0.2       , -8.23366899,
        0.3       ]), 'mass_matrix': np.array([-8.78260239,  0.1       , -8.37756698,  0.2       , -8.23367203,
        0.3       ]), 'scaled_mass': np.array([-8.78259981,  0.1       , -8.37757002,  0.2       , -8.23366946,
        0.3       ])}, 'lu': {'no_precond': np.array([-8.78259913,  0.1       , -8.37756829,  0.2       , -8.23366915,
        0.3       ]), 'mass_matrix': np.array([-8.78259957,  0.1       , -8.37756825,  0.2       , -8.23366863,
        0.3       ])}, 'muller_external': {'no_precond': np.array([-8.7905731 ,  0.1       , -8.38039315,  0.2       , -8.23488335,
        0.3       ]), 'mass_matrix': np.array([-8.7905552 ,  0.1       , -8.38040263,  0.2       , -8.23488802,
        0.3       ])}, 'muller_internal': {'no_precond': np.array([-8.79049006,  0.1       , -8.38036516,  0.2       , -8.23485991,
        0.3       ]), 'mass_matrix': np.array([-8.79049045,  0.1       , -8.3803654 ,  0.2       , -8.23485866,
        0.3       ])}}

"""
