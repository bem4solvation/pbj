import pytest
import pbj
import pbj.electrostatics.pb_formulation.formulations as pb_formulations
from inspect import getmembers, ismodule, isfunction
import numpy as np

values = {}
spheres = []

for formulation_name, object_address in getmembers(pb_formulations, ismodule):
    formulation = getattr(pb_formulations, formulation_name, None)
    values[formulation_name] = {}
    values[formulation_name]['no_precond'] = np.array([])
    for precond_name, object_address in getmembers(formulation, isfunction):
        if precond_name.endswith('preconditioner'):
            precond_name_removed = precond_name[:-15]
            if not precond_name_removed.startswith('calderon'):
                values[formulation_name][precond_name_removed] = np.array([])

print(values)

solvation_value = 8
tol = 1


for mesh_dens in [0.1,0.2,0.3]:
    sphere = pbj.Solute('test.pqr', mesh_density = mesh_dens, mesh_generator="msms")
    sphere.x_q[0][0]=0.1
    spheres.append(sphere)

# No preconditioner
for sphere in spheres:
    for formulation in values.keys():
        sphere.pb_formulation = formulation 
        for preconditioner in values[formulation].keys():
            if preconditioner == 'no_precond':
                sphere.pb_formulation_preconditioning = False
                sphere.calculate_solvation_energy(rerun_all = True)
                print(values[formulation]['no_precond'])
                print(sphere.results['solvation_energy'])
                print(sphere.mesh_density)
                values[formulation]['no_precond'] = np.append(values[formulation]['no_precond'],(sphere.results["solvation_energy"], sphere.mesh_density))
        
            else:
                sphere.pb_formulation_preconditioning = True
                sphere.pb_formulation_preconditioning_type = preconditioner
                sphere.calculate_solvation_energy(rerun_all = True)
                print("Current formulation: {}".format(formulation))
                print("Current sphere: {}".format(sphere.mesh_density))
                print("Current preconditioner: {}".format(preconditioner))
                values[formulation][preconditioner] = np.append(values[formulation][preconditioner],(sphere.results["solvation_energy"], sphere.mesh_density))
print(values)


solvation_energy_values = np.array([])
solvation_energy_expected_values = np.array([])
for formulation in values.keys():
    for preconditioner in values[formulation].keys():
        val_array = values[formulation][preconditioner]
        val = np.mean([val_array[0],val_array[2],val_array[4]])
        solvation_energy_values = np.append(solvation_energy_values,np.array([formulation,preconditioner,val]))
        solvation_energy_expected_values = np.append(solvation_energy_expected_values,np.array([formulation,preconditioner,solvation_value]))
        



"""
@pytest.mark.parametrize("test_input, expected", zip(solvation_energy_values, solvation_energy_expected_values))
def test_solvation_energy(test_input, expected):
  assert test_input[2] == expected[2]
@pytest.mark.parametrize("test_input, expected", zip(solvation_energy_values, solvation_energy_expected_values))
test_solvation_energy(test_input, expected)
"""