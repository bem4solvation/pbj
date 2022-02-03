import os
import numpy as np
import shutil
from .mesh_tools import (
    convert_pdb2pqr,
    generate_msms_mesh,
    convert_pqr2xyzr,
    generate_nanoshaper_mesh,
    import_msms_mesh,
)


def import_charges_from_pqr(pqr_path):
    """
    Given a pqr file, import the charges and the coordinates of them. Returns two arrays,
    q with the charges and x_q with the coordinates.

    Parameters
    ----------
    pqr_path : str
        Path to the pqr file.

    Returns
    -------
        q : numpy.array
            Array with the charges.
        x_q : numpy.array
            Array with the coordinates of the charges.

    Examples
    --------
    >>> import numpy as np
    >>> import pbj.mesh.charge_tools as ct
    >>> pqr_path = "methane.pqr"
    >>> q, x_q = ct.import_charges_from_pqr(pqr_path)
    >>> print(q)
    [-0.1048  0.0262  0.0262  0.0262  0.0262]
    >>> print(x_q)
    [[-0.683  0.813  0.254]
     [-0.129  1.613  0.75 ]
     [ 0.     0.    -0.   ]
     [-1.462  0.44   0.923]
     [-1.142  1.201 -0.658]]
    """
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
    """
    Generate mesh grid for the solute class given by parameter. It returns a grid class,
    and two arrays for the charges and the coordinates of the charges.

    Parameters
    ----------
    solute : class
        Solute class.

    Returns
    -------
        grid : class
            Bempp Grid object.
        q : numpy.array
            Array with the charges of the solute.
        x_q : numpy.array
            Array with the coordinates of the charges.
    """
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

    # Esta cuestionable, al parecer era codigo de bempp legacy, pero ahora no es necesario
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


def load_charges_to_solute(solute):
    """
    Given a class solute as a parameter, it loads the charges and the coordinates of the charges to the class,
    using the paths saved in the class and the function import_charges_from_pqr.

    Parameters
    ----------
        solute : class
            Solute class.

    Returns
    -------
        q : numpy.array
            Array with the charges of the solute.
        x_q : numpy.array
            Array with the coordinates of the charges.
    """

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
