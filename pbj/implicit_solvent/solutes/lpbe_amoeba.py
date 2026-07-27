from .solute_common import Solute
import pbj.implicit_solvent.pb_formulation.lpbe_amoeba as pb_formulations
import pbj.mesh.charge_tools as charge_tools
import pbj.mesh.mesh_tools as mesh_tools
import os
import bempp_cl as bempp
import numpy as np


class LPBE_AMOEBA(Solute):
    """LPBE_AMOEBA solute specialization."""

    def __init__(
        self,
        solute_file_path,
        external_mesh_file=None,
        solute_type="lpbe_amoeba",
        force_field="amoeba",
        radius_keyword="solute",
        solute_radius_type="PB",
        formulation="direct_amoeba",
        **kwargs,
    ):
        super().__init__(
            solute_file_path=solute_file_path,
            solute_type=solute_type,
            formulation=formulation,
            force_field=force_field,
            **kwargs,
        )

        self.formulation_object = getattr(pb_formulations, self.pb_formulation, None)
        if self.formulation_object is None:
            raise ValueError("Unrecognised formulation type %s" % self.pb_formulation)

        self.radius_keyword = radius_keyword
        self.solute_radius_type = solute_radius_type

        if external_mesh_file is not None:
            filename, file_extension = os.path.splitext(external_mesh_file)
            if file_extension == "":  # Assume use of vert and face
                self.external_mesh_face_path = external_mesh_file + ".face"
                self.external_mesh_vert_path = external_mesh_file + ".vert"
                self.mesh = mesh_tools.import_msms_mesh(
                    self.external_mesh_face_path, self.external_mesh_vert_path
                )

            else:  # Assume use of file that can be directly imported into bempp
                self.external_mesh_file_path = external_mesh_file
                self.mesh = bempp.api.import_grid(self.external_mesh_file_path)

                (
                    self.x_q,
                    self.q,
                    self.d,
                    self.Q,
                    self.alpha,
                    self.r_q,
                    self.mass,
                    self.polar_group,
                    self.thole,
                    self.connections_12,
                    self.connections_13,
                    self.pointer_connections_12,
                    self.pointer_connections_13,
                    self.p12scale,
                    self.p13scale,
                ) = charge_tools.load_tinker_multipoles_to_solute(self)

                self.d_induced = np.zeros_like(self.d)
                self.d_induced_prev = np.zeros_like(self.d)

        else:  # Generate mesh from given pdb or pqr, and import charges at the same time

            (
                self.mesh,
                self.x_q,
                self.q,
                self.d,
                self.Q,
                self.alpha,
                self.r_q,
                self.mass,
                self.polar_group,
                self.thole,
                self.connections_12,
                self.connections_13,
                self.pointer_connections_12,
                self.pointer_connections_13,
                self.p12scale,
                self.p13scale,
            ) = charge_tools.generate_msms_mesh_import_tinker_multipoles(self)

            self.d_induced = np.zeros_like(self.d)
            self.d_induced_prev = np.zeros_like(self.d)

        # Setup Dirichlet and Neumann spaces to use, save these as object vars
        dirichl_space = bempp.api.function_space(self.mesh, "P", 1)
        neumann_space = dirichl_space
        self.dirichl_space = dirichl_space
        self.neumann_space = neumann_space

    def calculate_solvation_energy(
        self,
        electrostatic_energy=True,
        nonpolar_energy=False,
        units="kcal_mol",
    ):
        print("Calculo no disponible para polarizable!")

    def calculate_solvation_forces(
        self,
        h=0.001,
        force_formulation="maxwell_tensor",
        fdb_approx=False,
        units="kcal_molA",
    ):
        print("Calculo no disponible para polarizable!")
