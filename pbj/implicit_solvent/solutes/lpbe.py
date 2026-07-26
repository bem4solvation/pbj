from .solute_common import Solute


class LPBE(Solute):
    """LPBE solute specialization."""

    def __init__(
        self,
        solute_file_path,
        **kwargs,
    ):
        super().__init__(solute_file_path=solute_file_path, **kwargs)
        self.solute_type = "lpbe"

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
