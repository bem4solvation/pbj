from .solute_common import Solute


class LPBE_AMOEBA(Solute):
    """LPBE_AMOEBA solute specialization."""

    def __init__(
        self,
        solute_file_path,
        **kwargs,
    ):
        super().__init__(solute_file_path=solute_file_path, **kwargs)
