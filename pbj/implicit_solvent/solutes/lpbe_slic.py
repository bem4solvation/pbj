from .lpbe import LPBE


class LPBE_SLIC(LPBE):
    """LPBE solute specialization."""

    def __init__(self, solute_file_path, external_mesh_file=None, **kwargs):
        """Initializes an LPBE solute prepared for the SLIC model.

        Behaves exactly like `LPBE.__init__`: it loads or generates the surface mesh and the
        atomic charges and creates the boundary element spaces. The SLIC-specific parameters
        (such as the Stern layer mesh) are configured later by the simulation.

        Args:
            solute_file_path (str): Path to the molecular structure file (.pdb, .pqr, or .xyz).
            external_mesh_file (str, optional): Path to a precomputed surface mesh. See
                `LPBE.__init__` for the supported formats. Defaults to None.
            **kwargs: Additional keyword arguments forwarded to `LPBE.__init__`.
        """

        super().__init__(
            solute_file_path=solute_file_path,
            external_mesh_file=external_mesh_file,
            **kwargs
        )
