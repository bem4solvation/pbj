from .lpbe import LPBE


class LPBE_SLIC(LPBE):
    """LPBE solute specialization."""

    def __init__(self, solute_file_path, external_mesh_file=None, **kwargs):

        super().__init__(
            solute_file_path=solute_file_path,
            external_mesh_file=external_mesh_file,
            **kwargs
        )
