import re
import bempp.api
import os
import numpy as np
import time
import pbj.mesh.mesh_tools as mesh_tools
import pbj.mesh.charge_tools as charge_tools
import pbj.electrostatics.pb_formulation.formulations as pb_formulations
import pbj.electrostatics.utils as utils


class Solute:
    """The basic Solute object
    This object holds all the solute information and allows for an easy way to hold the data
    """

    def __init__(
        self,
        solute_file_path,
        external_mesh_file=None,
        save_mesh_build_files=False,
        mesh_build_files_dir="mesh_files/",
        mesh_density=1.0,
        nanoshaper_grid_scale=None,
        mesh_probe_radius=1.4,
        mesh_generator="nanoshaper",
        print_times=False,
        force_field="amber",
        formulation="direct",
    ):

        if not os.path.isfile(solute_file_path):
            print("file does not exist -> Cannot start")
            return

        self._pb_formulation = formulation

        self.formulation_object = getattr(pb_formulations, self.pb_formulation, None)
        if self.formulation_object is None:
            raise ValueError("Unrecognised formulation type %s" % self.pb_formulation)

        self.force_field = force_field

        self.save_mesh_build_files = save_mesh_build_files
        self.mesh_build_files_dir = os.path.abspath(mesh_build_files_dir)

        if nanoshaper_grid_scale is not None:
            if mesh_generator == "nanoshaper":
                print("Using specified grid_scale.")
                self.nanoshaper_grid_scale = nanoshaper_grid_scale
            else:
                print(
                    "Ignoring specified grid scale as mesh_generator is not specified as nanoshaper."
                )
                self.mesh_density = mesh_density
        else:
            self.mesh_density = mesh_density
            if mesh_generator == "nanoshaper":
                self.nanoshaper_grid_scale = (
                    mesh_tools.density_to_nanoshaper_grid_scale_conversion(
                        self.mesh_density
                    )
                )
        self.mesh_probe_radius = mesh_probe_radius
        self.mesh_generator = mesh_generator

        self.print_times = print_times

        file_extension = solute_file_path.split(".")[-1]
        if file_extension == "pdb":
            self.imported_file_type = "pdb"
            self.pdb_path = solute_file_path
            self.solute_name = get_name_from_pdb(self.pdb_path)

        elif file_extension == "pqr":
            self.imported_file_type = "pqr"
            self.pqr_path = solute_file_path
            self.solute_name = os.path.split(solute_file_path.split(".")[-2])[-1]

        else:
            print("File is not pdb or pqr -> Cannot start")

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

            self.q, self.x_q, self.r_q = charge_tools.load_charges_to_solute(
                self
            )  # Import charges from given file

        else:  # Generate mesh from given pdb or pqr, and import charges at the same time
            (
                self.mesh,
                self.q,
                self.x_q,
                self.r_q
            ) = charge_tools.generate_msms_mesh_import_charges(self)

        self.ep_in = 4.0
        self.ep_ex = 80.0
        self.kappa = 0.125

        self.pb_formulation_alpha = 1.0  # np.nan
        self.pb_formulation_beta = self.ep_ex / self.ep_in  # np.nan

        self.pb_formulation_stern_width = 2.0

        self.slic_alpha = 0.5
        self.slic_beta = -60
        self.slic_gamma = -0.5

        self.slic_max_iterations = 20
        self.slic_tolerance = 1e-5

        self.pb_formulation_preconditioning = False
        self.pb_formulation_preconditioning_type = "calderon_squared"

        self.discrete_form_type = "strong"

        self.gmres_tolerance = 1e-5
        self.gmres_restart = 1000
        self.gmres_max_iterations = 1000

        self.operator_assembler = "dense"
        self.rhs_constructor = "numpy"

        self.matrices = dict()
        self.rhs = dict()
        self.results = dict()
        self.timings = dict()

        # Setup Dirichlet and Neumann spaces to use, save these as object vars
        dirichl_space = bempp.api.function_space(self.mesh, "P", 1)
        # neumann_space = bempp.api.function_space(self.mesh, "P", 1)
        neumann_space = dirichl_space
        self.dirichl_space = dirichl_space
        self.neumann_space = neumann_space

    @property
    def pb_formulation(self):
        return self._pb_formulation

    @pb_formulation.setter
    def pb_formulation(self, value):
        self._pb_formulation = value
        self.formulation_object = getattr(pb_formulations, self.pb_formulation, None)
        if self.formulation_object is None:
            raise ValueError("Unrecognised formulation type %s" % self.pb_formulation)

    def display_available_formulations(self):
        from inspect import getmembers, ismodule

        print("Current formulation: " + self.pb_formulation)
        print("List of available formulations:")
        available = getmembers(pb_formulations, ismodule)
        for element in available:
            if element[0] == 'common':
                available.remove(element)
        for name, object_address in available:
            print(name)

    def display_available_preconditioners(self):
        from inspect import getmembers, isfunction

        print(
            "List of preconditioners available for the current formulation ("
            + self.pb_formulation
            + "):"
        )
        for name, object_address in getmembers(self.formulation_object, isfunction):
            if name.endswith("preconditioner"):
                name_removed = name[:-15]
                print(name_removed)

    def initialise_matrices(self):
        start_time = time.time()  # Start the timing for the matrix construction
        # Construct matrices based on the desired formulation
        # Verify if parameters are already set and save A matrix
        if self.formulation_object.verify_parameters(self):
            self.formulation_object.lhs(self)
        self.timings["time_matrix_initialisation"] = time.time() - start_time

    def assemble_matrices(self):
        start_assembly = time.time()
        self.matrices["A"].weak_form()
        self.timings["time_matrix_assembly"] = time.time() - start_assembly

    def initialise_rhs(self):
        start_rhs = time.time()
        # Verify if parameters are already set and then save RHS
        if self.formulation_object.verify_parameters(self):
            self.formulation_object.rhs(self)
        self.timings["time_rhs_initialisation"] = time.time() - start_rhs

    def apply_preconditioning(self):
        preconditioning_start_time = time.time()
        if self.pb_formulation_preconditioning:
            precon_str = self.pb_formulation_preconditioning_type + "_preconditioner"
            preconditioning_object = getattr(self.formulation_object, precon_str, None)
            if preconditioning_object is not None:
                preconditioning_object(self)
            else:
                raise ValueError(
                    "Unrecognised preconditioning type %s for current formulation type %s"
                    % (self.pb_formulation_preconditioning_type, self.pb_formulation)
                )
        else:
            self.matrices["A_final"] = self.matrices["A"]
            self.rhs["rhs_final"] = [rhs for key,rhs in sorted(self.rhs.items())][:len(self.matrices["A"].domain_spaces)]

            self.matrices["A_discrete"] = matrix_to_discrete_form(
                self.matrices["A_final"], "weak"
            )
            self.rhs["rhs_discrete"] = rhs_to_discrete_form(
                self.rhs["rhs_final"], "weak", self.matrices["A"]
            )

        self.timings["time_preconditioning"] = time.time() - preconditioning_start_time

    def calculate_potential(self, rerun_all=False):
        if self.formulation_object.verify_parameters(self):
            self.formulation_object.calculate_potential(self, rerun_all)


    def calculate_solvation_energy(self, rerun_all=False):
        if rerun_all:
            self.calculate_potential(rerun_all)

        if "phi" not in self.results:
            # If surface potential has not been calculated, calculate it now
            self.calculate_potential()

        start_time = time.time()

        solution_dirichl = self.results["phi"]
        solution_neumann = self.results["d_phi"]

        from bempp.api.operators.potential.laplace import single_layer, double_layer

        slp_q = single_layer(self.neumann_space, self.x_q.transpose())
        dlp_q = double_layer(self.dirichl_space, self.x_q.transpose())
        phi_q = slp_q * solution_neumann - dlp_q * solution_dirichl

        # total solvation energy applying constant to get units [kcal/mol]
        total_energy = 2 * np.pi * 332.064 * np.sum(self.q * phi_q).real
        self.results["solvation_energy"] = total_energy
        self.timings["time_calc_energy"] = time.time() - start_time

        if self.print_times:
            print(
                "It took ",
                self.timings["time_calc_energy"],
                " seconds to compute the solvation energy",
            )


def get_name_from_pdb(pdb_path):
    pdb_file = open(pdb_path)
    first_line = pdb_file.readline()
    first_line_split = re.split(r"\s{2,}", first_line)
    solute_name = first_line_split[3].lower()
    pdb_file.close()

    return solute_name


def matrix_to_discrete_form(matrix, discrete_form_type):
    if discrete_form_type == "strong":
        matrix_discrete = matrix.strong_form()
    elif discrete_form_type == "weak":
        matrix_discrete = matrix.weak_form()
    else:
        raise ValueError("Unexpected discrete type: %s" % discrete_form_type)

    return matrix_discrete


def rhs_to_discrete_form(rhs_list, discrete_form_type, A):
    from bempp.api.assembly.blocked_operator import (
        coefficients_from_grid_functions_list,
        projections_from_grid_functions_list,
    )

    if discrete_form_type == "strong":
        rhs = coefficients_from_grid_functions_list(rhs_list)
    elif discrete_form_type == "weak":
        rhs = projections_from_grid_functions_list(rhs_list, A.dual_to_range_spaces)
    else:
        raise ValueError("Unexpected discrete form: %s" % discrete_form_type)

    return rhs


