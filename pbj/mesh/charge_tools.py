import os 
#import .mesh_tools
from .mesh_tools import *



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
        convert_pdb2pqr(solute.pdb_path,
                                   mesh_pqr_path,
                                   solute.force_field
                                   )
    else:
        mesh_pqr_path = solute.pqr_path

    mesh_xyzr_path = os.path.join(mesh_dir, solute.solute_name + ".xyzr")
    convert_pqr2xyzr(mesh_pqr_path, mesh_xyzr_path)

    mesh_face_path = os.path.join(mesh_dir, solute.solute_name + ".face")
    mesh_vert_path = os.path.join(mesh_dir, solute.solute_name + ".vert")

    if solute.mesh_generator == "msms":
        generate_msms_mesh(mesh_xyzr_path,
                                      mesh_dir,
                                      solute.solute_name,
                                      solute.mesh_density,
                                      solute.mesh_probe_radius
                                      )
    elif solute.mesh_generator == "nanoshaper":
        generate_nanoshaper_mesh(mesh_xyzr_path,
                                            mesh_dir,
                                            solute.solute_name,
                                            solute.nanoshaper_grid_scale,
                                            solute.mesh_probe_radius,
                                            solute.save_mesh_build_files
                                            )

    mesh_off_path = os.path.join(mesh_dir, solute.solute_name + ".off")
    
    #### Esta cuestionable
    # convert_msms2off(mesh_face_path, mesh_vert_path, mesh_off_path)

    grid = import_msms_mesh(mesh_face_path, mesh_vert_path)
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
        convert_pdb2pqr(solute.pdb_path, mesh_pqr_path, solute.force_field)
    else:
        mesh_pqr_path = solute.pqr_path

    q, x_q = utils.import_charges(mesh_pqr_path)

    return q, x_q
