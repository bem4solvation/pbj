import bempp.api
import time

# import numpy as np
import pbj.electrostatics.solute
import pbj.electrostatics.pb_formulation.formulations as pb_formulations
import pbj.electrostatics.utils as utils


class Simulation:
    def __init__(self, formulation="direct"):

        self._pb_formulation = formulation
        self.formulation_object = getattr(pb_formulations, self.pb_formulation, None)
        if self.formulation_object is None:
            raise ValueError("Unrecognised formulation type %s" % self.pb_formulation)

        self.solvent_parameters = dict()
        self.solvent_parameters["ep"] = 80.0

        self.gmres_tolerance = 1e-5
        self.gmres_restart = 1000
        self.gmres_max_iterations = 1000

        self.solutes = list()
        self.matrices = dict()
        self.rhs = dict()
        self.timings = dict()

    @property
    def pb_formulation(self):
        return self._pb_formulation

    @pb_formulation.setter
    def pb_formulation(self, value):
        self._pb_formulation = value
        self.formulation_object = getattr(pb_formulations, self.pb_formulation, None)
        self.matrices["preconditioning_matrix_gmres"] = None
        if self.formulation_object is None:
            raise ValueError("Unrecognised formulation type %s" % self.pb_formulation)

    def add_solute(self, solute):

        if isinstance(solute, pbj.electrostatics.solute.Solute):
            if solute in self.solutes:
                print("Solute object is already added to this simulation. Ignoring this add command.")
            else:
                self.solutes.append(solute)
        else:
            raise ValueError("Given object is not of the 'Solute' class.")

    def create_and_assemble_matrix(self):
        surface_count = len(self.solutes)
        A = bempp.api.BlockedOperator(surface_count, surface_count)

        # Get self interactions of each solute
        for index, solute in enumerate(self.solutes):
            solute.pb_formulation = self.pb_formulation
            solute.pb_formulation_preconditioning = False

            solute.initialise_matrices()
            solute.initialise_rhs()
            solute.assemble_matrices()
            solute.apply_preconditioning()

            A[index, index] = solute.matrices["A_discrete"]
            self.rhs["rhs_" + str((index * 2) + 1)] = solute.rhs["rhs_1"]
            self.rhs["rhs_" + str((index * 2) + 2)] = solute.rhs["rhs_2"]

        # Calculate matrix elements for interactions between solutes
        for i in range(surface_count):
            for j in range(surface_count):
                if i == j:
                    continue
                else:
                    A[i, j] = self.formulation_object.inter_solute_interactions()

        self.matrices["A_discrete"] = A

    def calculate_potentials(self):
        self.create_and_assemble_matrix()

        # Use GMRES to solve the system of equations
        gmres_start_time = time.time()
        x, info, it_count = utils.solver(
            self.matrices["A_discrete"],
            self.rhs["rhs_discrete"],
            self.gmres_tolerance,
            self.gmres_restart,
            self.gmres_max_iterations,
        )

        self.timings["time_gmres"] = time.time() - gmres_start_time
