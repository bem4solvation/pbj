from pbj.implicit_solvent import Solute


class LPBE_AMOEBA(Solute):
    """The basic Solute object
    This object holds all the solute information and allows for an easy way to hold the data
    """

    def __init__(
        self,
        solute_file_path,
        external_mesh_file=None,
        save_mesh_build_files=False,
        mesh_build_files_dir="mesh_files/",
        mesh_density=2.0,
        nanoshaper_grid_scale=None,
        solvent_radius=1.4,
        mesh_generator="nanoshaper",
        print_times=False,
        force_field="amber",
        formulation="direct",
        radius_keyword="solute",
        solute_radius_type="PB",
        fill_cavities=True,
        cavity_cutoff=60,
    ):
        super().__init__(
            solute_file_path=solute_file_path,
            external_mesh_file=external_mesh_file,
            save_mesh_build_files=save_mesh_build_files,
            mesh_build_files_dir=mesh_build_files_dir,
            mesh_density=mesh_density,
            nanoshaper_grid_scale=nanoshaper_grid_scale,
            solvent_radius=solvent_radius,
            mesh_generator=mesh_generator,
            print_times=print_times,
            force_field=force_field,
            formulation=formulation,
            radius_keyword=radius_keyword,
            solute_radius_type=solute_radius_type,
            fill_cavities=fill_cavities,
            cavity_cutoff=cavity_cutoff,
        )
