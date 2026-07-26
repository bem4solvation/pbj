from .solute_common import Solute
import pbj.implicit_solvent.pb_formulation.lpbe_amoeba as pb_formulations
import pbj.mesh.charge_tools as charge_tools
import pbj.mesh.mesh_tools as mesh_tools
import os
import bempp_cl as bempp


class LPBE(Solute):
    """LPBE solute specialization."""

    def __init__(
        self,
        solute_file_path,
        formulation="direct",
        **kwargs,
    ):
        super().__init__(solute_file_path=solute_file_path, **kwargs)

        self._pb_formulation = formulation

        self.formulation_object = getattr(pb_formulations, self.pb_formulation, None)
        if self.formulation_object is None:
            raise ValueError("Unrecognised formulation type %s" % self.pb_formulation)

        if kwargs.get("external_mesh_file") is not None:
            filename, file_extension = os.path.splitext(kwargs["external_mesh_file"])
            if file_extension == "":  # Assume use of vert and face
                self.external_mesh_face_path = kwargs["external_mesh_file"] + ".face"
                self.external_mesh_vert_path = kwargs["external_mesh_file"] + ".vert"
                self.mesh = mesh_tools.import_msms_mesh(
                    self.external_mesh_face_path, self.external_mesh_vert_path
                )

            else:  # Assume use of file that can be directly imported into bempp
                self.external_mesh_file_path = kwargs["external_mesh_file"]
                self.mesh = bempp.api.import_grid(self.external_mesh_file_path)

            (
                self.q,
                self.x_q,
                self.r_q,
                self.atom_name,
                self.res_name,
                self.res_num,
            ) = charge_tools.load_charges_to_solute(
                self
            )  # Import charges from given file

        else:  # Generate mesh from given pdb or pqr, and import charges at the same time

            (
                self.mesh,
                self.q,
                self.x_q,
                self.r_q,
                self.atom_name,
                self.res_name,
                self.res_num,
            ) = charge_tools.generate_msms_mesh_import_charges(self)

    def calculate_solvation_energy(
        self,
        electrostatic_energy=True,
        nonpolar_energy=False,
        units="kcal_mol",
    ):
        print("Calculando para LPBE!")
        super().calculate_solvation_energy(
            electrostatic_energy=electrostatic_energy,
            nonpolar_energy=nonpolar_energy,
            units=units,
        )
