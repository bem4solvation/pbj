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
                print(
                    "Solute object is already added to this simulation. Ignoring this add command."
                )
            else:
                self.solutes.append(solute)
        else:
            raise ValueError("Given object is not of the 'Solute' class.")

    def create_and_assemble_matrix(self):
        surface_count = len(self.solutes)
        A = bempp.api.BlockedOperator(surface_count * 2, surface_count * 2)
        diag_block_indexes = []

        # Get self interactions of each solute
        for index, solute in enumerate(self.solutes):
            solute.pb_formulation = self.pb_formulation
            solute.pb_formulation_preconditioning = False

            solute.initialise_matrices()
            solute.initialise_rhs()
            solute.assemble_matrices()

            A[index * 2, index * 2] = solute.matrices["A"][0, 0]
            A[(index * 2) + 1, index * 2] = solute.matrices["A"][1, 0]
            A[index * 2, (index * 2) + 1] = solute.matrices["A"][0, 1]
            A[(index * 2) + 1, (index * 2) + 1] = solute.matrices["A"][1, 1]

            self.rhs["rhs_" + str(index + 1)] = [
                solute.rhs["rhs_1"],
                solute.rhs["rhs_2"],
            ]

            diag_block_indexes.extend(
                [
                    [index * 2, index * 2],
                    [(index * 2) + 1, index * 2],
                    [index * 2, (index * 2) + 1],
                    [(index * 2) + 1, (index * 2) + 1],
                ]
            )

        # Calculate matrix elements for interactions between solutes
        for i in range(surface_count * 2):
            for j in range(surface_count * 2):
                if [i, j] in diag_block_indexes:
                    continue
                # else: ##NEEDS MORE WORK TO IMPLEMENT
                #     A[i, j] = self.formulation_object.inter_solute_interactions(
                #         self, i, j
                #     )

        self.matrices["A"] = A

    def apply_preconditioning(self):
        self.matrices["A_final"] = self.matrices["A"]

        rhs_final = []
        for solute_rhs in self.rhs:
            rhs_final.extend(solute_rhs)

        self.rhs["rhs_final"] = rhs_final

        self.matrices["A_discrete"] = utils.matrix_to_discrete_form(
            self.matrices["A_final"], "weak"
        )
        self.rhs["rhs_discrete"] = utils.rhs_to_discrete_form(
            self.rhs["rhs_final"], "weak", self.matrices["A"]
        )

    def calculate_potentials(self):
        self.create_and_assemble_matrix()
        self.apply_preconditioning()

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
