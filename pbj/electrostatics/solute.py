import re
import bempp.api
import os
import numpy as np
import time
import pbj.mesh.mesh_tools as mesh_tools
#import bem_electrostatics.pb_formulation as pb_formulation
#import bem_electrostatics.utils as utils


class Solute:
    """The basic Solute object
    This object holds all the solute information and allows for an easy way to hold the data"""

    def __init__(self,
                 solute_file_path,
                 external_mesh_file=None,
                 save_mesh_build_files=False,
                 mesh_build_files_dir="mesh_files/",
                 mesh_density=1.0,
                 nanoshaper_grid_scale=None,
                 mesh_probe_radius=1.4,
                 mesh_generator="nanoshaper",
                 print_times=False,
                 force_field="amber"
                 ):

        if not os.path.isfile(solute_file_path):
            print("file does not exist -> Cannot start")
            return

        self.force_field = force_field

        self.save_mesh_build_files = save_mesh_build_files
        self.mesh_build_files_dir = os.path.abspath(mesh_build_files_dir)

        if nanoshaper_grid_scale is not None:
            if mesh_generator == 'nanoshaper':
                print('Using specified grid_scale.')
                self.nanoshaper_grid_scale = nanoshaper_grid_scale
            else:
                print('Ignoring specified grid scale as mesh_generator is not specified as nanoshaper.')
                self.mesh_density = mesh_density
        else:
            self.mesh_density = mesh_density
            if mesh_generator == 'nanoshaper':
                self.nanoshaper_grid_scale = mesh_tools.density_to_nanoshaper_grid_scale_conversion(
                    self.mesh_density)
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
            self.solute_name = solute_file_path.split(".")[-2].split("/")[-1]

        else:
            print("File is not pdb or pqr -> Cannot start")
        """

        if external_mesh_file is not None:
            filename, file_extension = os.path.splitext(external_mesh_file)
            if file_extension == "":  # Assume use of vert and face
                self.external_mesh_face_path = external_mesh_file + ".face"
                self.external_mesh_vert_path = external_mesh_file + ".vert"
                self.mesh = mesh_tools.import_msms_mesh(self.external_mesh_face_path, self.external_mesh_vert_path)

            else:  # Assume use of file that can be directly imported into bempp
                self.external_mesh_file_path = external_mesh_file
                self.mesh = bempp.api.import_grid(self.external_mesh_file_path)

            self.q, self.x_q = import_charges(self)  # Import charges from given file

        else:  # Generate mesh from given pdb or pqr, and import charges at the same time
            self.mesh, self.q, self.x_q = generate_msms_mesh_import_charges(self)

        self.pb_formulation = "direct"
        """
        self.ep_in = 4.0
        self.ep_ex = 80.0
        self.kappa = 0.125

        self.pb_formulation_alpha = 1.0
        self.pb_formulation_beta = self.ep_ex / self.ep_in

        self.pb_formulation_preconditioning = False
        self.pb_formulation_preconditioning_type = "calderon_squared"

        self.discrete_form_type = "strong"

        self.gmres_tolerance = 1e-5
        self.gmres_restart = 1000
        self.gmres_max_iterations = 1000

        self.operator_assembler = 'dense'
        self.rhs_constructor = 'numpy'

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

################################
################################
# Bloque para funciones 
# initialise_matrices
# assemble_matrices
# initialise_rhs
# apply_preconditioning
# pass_to_discrete_form

# calculate_potential
# calculate_solvation_energy

########################
"""
def generate_msms_mesh_import_charges(solute):
    mesh_dir = os.path.abspath("mesh_temp/")
    if solute.save_mesh_build_files:
        mesh_dir = solute.mesh_build_files_dir

    if not os.path.exists(mesh_dir):
        try:
            os.mkdir(mesh_dir)
        except OSError:
            print("Creation of the directory %s failed" % mesh_dir)

    if solute.imported_file_type == "pdb":
        mesh_pqr_path = os.path.join(mesh_dir, solute.solute_name + ".pqr")
        mesh_pqr_log = os.path.join(mesh_dir, solute.solute_name + ".log")
        mesh_tools.convert_pdb2pqr(solute.pdb_path,
                                   mesh_pqr_path,
                                   solute.force_field
                                   )
    else:
        mesh_pqr_path = solute.pqr_path

    mesh_xyzr_path = os.path.join(mesh_dir, solute.solute_name + ".xyzr")
    mesh_tools.convert_pqr2xyzr(mesh_pqr_path, mesh_xyzr_path)

    mesh_face_path = os.path.join(mesh_dir, solute.solute_name + ".face")
    mesh_vert_path = os.path.join(mesh_dir, solute.solute_name + ".vert")

    if solute.mesh_generator == "msms":
        mesh_tools.generate_msms_mesh(mesh_xyzr_path,
                                      mesh_dir,
                                      solute.solute_name,
                                      solute.mesh_density,
                                      solute.mesh_probe_radius
                                      )
    elif solute.mesh_generator == "nanoshaper":
        mesh_tools.generate_nanoshaper_mesh(mesh_xyzr_path,
                                            mesh_dir,
                                            solute.solute_name,
                                            solute.nanoshaper_grid_scale,
                                            solute.mesh_probe_radius,
                                            solute.save_mesh_build_files
                                            )

    mesh_off_path = os.path.join(mesh_dir, solute.solute_name + ".off")
    
    #### Esta cuestionable
    # mesh_tools.convert_msms2off(mesh_face_path, mesh_vert_path, mesh_off_path)

    grid = mesh_tools.import_msms_mesh(mesh_face_path, mesh_vert_path)
    q, x_q = utils.import_charges(mesh_pqr_path)

    if solute.save_mesh_build_files:
        if solute.imported_file_type == "pdb":
            solute.mesh_pqr_path = mesh_pqr_path
            solute.mesh_pqr_log = mesh_pqr_log
        solute.mesh_xyzr_path = mesh_xyzr_path
        solute.mesh_face_path = mesh_face_path
        solute.mesh_vert_path = mesh_vert_path
        solute.mesh_off_path = mesh_off_path
    else:
        if solute.imported_file_type == "pdb":
            os.remove(mesh_pqr_path)
            os.remove(mesh_pqr_log)
        os.remove(mesh_xyzr_path)
        os.remove(mesh_face_path)
        os.remove(mesh_vert_path)
        os.remove(mesh_off_path)
        os.rmdir(mesh_dir)

    return grid, q, x_q


def import_charges(solute):
    mesh_dir = os.path.abspath("mesh_temp/")
    if solute.save_mesh_build_files:
        mesh_dir = solute.mesh_build_files_dir

    if not os.path.exists(mesh_dir):
        try:
            os.mkdir(mesh_dir)
        except OSError:
            print("Creation of the directory %s failed" % mesh_dir)

    if solute.imported_file_type == "pdb":
        mesh_pqr_path = os.path.join(mesh_dir, solute.solute_name + ".pqr")
        mesh_tools.convert_pdb2pqr(solute.pdb_path, mesh_pqr_path, solute.force_field)
    else:
        mesh_pqr_path = solute.pqr_path

    q, x_q = utils.import_charges(mesh_pqr_path)

    return q, x_q

"""
def get_name_from_pdb(pdb_path):
    pdb_file = open(pdb_path)
    first_line = pdb_file.readline()
    first_line_split = re.split(r'\s{2,}', first_line)
    solute_name = first_line_split[3].lower()
    pdb_file.close()

    return solute_name


def matrix_to_discrete_form(matrix, discrete_form_type):
    if discrete_form_type == "strong":
        matrix_discrete = matrix.strong_form()
    elif discrete_form_type == "weak":
        matrix_discrete = matrix.weak_form()
    else:
        raise ValueError('Unexpected discrete type: %s' % discrete_form_type)

    return matrix_discrete


def rhs_to_discrete_form(rhs_list, discrete_form_type, A):
    from bempp.api.assembly.blocked_operator import coefficients_from_grid_functions_list, \
        projections_from_grid_functions_list

    if discrete_form_type == "strong":
        rhs = coefficients_from_grid_functions_list(rhs_list)
    elif discrete_form_type == "weak":
        rhs = projections_from_grid_functions_list(rhs_list, A.dual_to_range_spaces)
    else:
        raise ValueError('Unexpected discrete form: %s' % discrete_form_type)

    return rhs


def show_potential_calculation_times(self):
    if "phi" in self.results:
        print('It took ', self.timings["time_matrix_construction"],
              ' seconds to construct the matrices')
        print('It took ', self.timings["time_rhs_construction"],
              ' seconds to construct the rhs vectors')
        print('It took ', self.timings["time_matrix_to_discrete"],
              ' seconds to pass the main matrix to discrete form (' + self.discrete_form_type + ')')
        print('It took ', self.timings["time_preconditioning"],
              ' seconds to compute and apply the preconditioning (' + str(self.pb_formulation_preconditioning)
              + '(' + self.pb_formulation_preconditioning_type + ')')
        print('It took ', self.timings["time_gmres"], ' seconds to resolve the system using GMRES')
        print('It took ', self.timings["time_compute_potential"], ' seconds in total to compute the surface potential')
    else:
        print('Potential must first be calculated to show times.')
    