import os

# import .mesh_tools
from .mesh_tools import *

def import_charges_from_pqr(pqr_path):
    # Read charges and coordinates from the .pqr file
    molecule_file = open(pqr_path, "r")
    molecule_data = molecule_file.read().split("\n")
    atom_count = 0
    for line in molecule_data:
        line = line.split()
        if len(line) == 0 or line[0] != "ATOM":
            continue
        atom_count += 1

    q, x_q = np.empty((atom_count,)), np.empty((atom_count, 3))
    count = 0
    for line in molecule_data:
        line = line.split()
        if len(line) == 0 or line[0] != "ATOM":
            continue
        q[count] = float(line[8])
        x_q[count, :] = line[5:8]
        count += 1

    return q, x_q


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
        convert_pdb2pqr(solute.pdb_path, mesh_pqr_path, solute.force_field)
    else:
        mesh_pqr_path = solute.pqr_path

    mesh_xyzr_path = os.path.join(mesh_dir, solute.solute_name + ".xyzr")
    convert_pqr2xyzr(mesh_pqr_path, mesh_xyzr_path)

    mesh_face_path = os.path.join(mesh_dir, solute.solute_name + ".face")
    mesh_vert_path = os.path.join(mesh_dir, solute.solute_name + ".vert")

    if solute.mesh_generator == "msms":
        generate_msms_mesh(
            mesh_xyzr_path,
            mesh_dir,
            solute.solute_name,
            solute.mesh_density,
            solute.mesh_probe_radius,
        )
    elif solute.mesh_generator == "nanoshaper":
        generate_nanoshaper_mesh(
            mesh_xyzr_path,
            mesh_dir,
            solute.solute_name,
            solute.nanoshaper_grid_scale,
            solute.mesh_probe_radius,
            solute.save_mesh_build_files,
        )

    mesh_off_path = os.path.join(mesh_dir, solute.solute_name + ".off")

    #### Esta cuestionable
    # convert_msms2off(mesh_face_path, mesh_vert_path, mesh_off_path)

    grid = import_msms_mesh(mesh_face_path, mesh_vert_path)
    q, x_q = import_charges_from_pqr(mesh_pqr_path)

    if solute.save_mesh_build_files:
        if solute.imported_file_type == "pdb":
            solute.mesh_pqr_path = mesh_pqr_path
            solute.mesh_pqr_log = mesh_pqr_log
        solute.mesh_xyzr_path = mesh_xyzr_path
        solute.mesh_face_path = mesh_face_path
        solute.mesh_vert_path = mesh_vert_path
        solute.mesh_off_path = mesh_off_path
    else:
        shutil.rmtree(mesh_dir)
    return grid, q, x_q


#
def load_charges_to_solute(solute):
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
        convert_pdb2pqr(solute.pdb_path, mesh_pqr_path, solute.force_field)
    else:
        mesh_pqr_path = solute.pqr_path

    q, x_q = import_charges_from_pqr(mesh_pqr_path)

    return q, x_q
