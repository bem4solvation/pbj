import os
import numpy as np
import shutil
from .mesh_tools import (
    convert_pdb2pqr,
    generate_msms_mesh,
    convert_pqr2xyzr,
    generate_nanoshaper_mesh,
    import_msms_mesh,
    fix_mesh,
    check_cavity,
)


def import_charges_from_pqr(pqr_path):
    """Imports the charges, coordinates, radii, and residue metadata from a PQR file.

    Args:
        pqr_path (str): Path to the PQR file.

    Returns:
        tuple: A tuple containing six elements:
            - q (numpy.ndarray): 1-dim array of atom charges.
            - x_q (numpy.ndarray): 2-dim array of shape $(N, 3)$ with the Cartesian coordinates of the charges.
            - r_q (numpy.ndarray): 1-dim array of atom radii.
            - atom_name (numpy.ndarray): 1-dim array of strings containing atom names.
            - res_name (numpy.ndarray): 1-dim array of strings containing residue names.
            - res_num (numpy.ndarray): 1-dim array of strings containing residue sequence numbers.

    Examples:
        >>> import numpy as np
        >>> import pbj.mesh.charge_tools as ct
        >>> pqr_path = "methane.pqr"
        >>> q, x_q, r_q, atom_name, res_name, res_num = ct.import_charges_from_pqr(pqr_path)
        >>> print(q)
        [-0.1048  0.0262  0.0262  0.0262  0.0262]
        >>> print(x_q)
        [[-0.683  0.813  0.254]
         [-0.129  1.613  0.75 ]
         [ 0.     0.    -0.   ]
         [-1.462  0.44   0.923]
         [-1.142  1.201 -0.658]]
        >>> print(r_q)
        [0.0262 0.0262 0.0262 0.0262 0.0262]
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

    q, x_q, r_q, atom_name, res_name, res_num = (
        np.empty((atom_count,)),
        np.empty((atom_count, 3)),
        np.empty((atom_count,)),
        np.empty((atom_count,), dtype=object),
        np.empty((atom_count,), dtype=object),
        np.empty((atom_count,), dtype=object),
    )
    count = 0
    for line in molecule_data:
        line = line.split()
        if len(line) == 0 or line[0] != "ATOM":
            continue
        q[count] = float(line[8])
        x_q[count, :] = line[5:8]
        r_q[count] = float(line[9])
        atom_name[count] = line[2]
        res_name[count] = line[3]
        res_num[count] = line[4]

        count += 1

    return q, x_q, r_q, atom_name, res_name, res_num


def generate_msms_mesh_import_charges(solute):
    """Generates the molecular mesh grid and imports charges for the given solute object.

    Handles file format conversions (PDB to PQR if needed, then to XYZR), runs the
    specified mesh generator (MSMS or NanoShaper), and extracts all charge and residue
    metadata.

    Args:
        solute (Solute): An instance of the Solute class containing the molecule's
            configuration parameters.

    Returns:
        tuple: A tuple containing seven elements:
            - grid (bempp.api.Grid): The imported Bempp Grid object.
            - q (numpy.ndarray): 1-dim array of atom charges.
            - x_q (numpy.ndarray): 2-dim array of shape $(N, 3)$ with the Cartesian coordinates of the charges.
            - r_q (numpy.ndarray): 1-dim array of atom radii.
            - atom_name (numpy.ndarray): 1-dim array of strings containing atom names.
            - res_name (numpy.ndarray): 1-dim array of strings containing residue names.
            - res_num (numpy.ndarray): 1-dim array of strings containing residue sequence numbers.
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
            cavity_cutoff=solute.cavity_cutoff,
            fill_cavities=solute.fill_cavities,
        )

    mesh_off_path = os.path.join(mesh_dir, solute.solute_name + ".off")

    grid = import_msms_mesh(mesh_face_path, mesh_vert_path)
    q, x_q, r_q, atom_name, res_name, res_num = import_charges_from_pqr(mesh_pqr_path)

    grid = check_cavity(
        grid, fill_cavities=solute.fill_cavities, volume_cutoff=solute.cavity_cutoff
    )
    if solute.mesh_generator == "msms":
        grid = fix_mesh(grid)

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
    return grid, q, x_q, r_q, atom_name, res_name, res_num


def load_charges_to_solute(solute):
    """Loads the charges, coordinates, and residue metadata for the given solute object.

    Handles file format conversions (PDB to PQR if needed) based on the solute's
    configuration, and extracts all charge and structural metadata using the
    corresponding PQR file.

    Args:
        solute (Solute): An instance of the Solute class containing the molecule's
            configuration parameters and file paths.

    Returns:
        tuple: A tuple containing six elements:
            - q (numpy.ndarray): 1-dim array of atom charges.
            - x_q (numpy.ndarray): 2-dim array of shape $(N, 3)$ with the Cartesian coordinates of the charges.
            - r_q (numpy.ndarray): 1-dim array of atom radii.
            - atom_name (numpy.ndarray): 1-dim array of strings containing atom names.
            - res_name (numpy.ndarray): 1-dim array of strings containing residue names.
            - res_num (numpy.ndarray): 1-dim array of strings containing residue sequence numbers.
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

    q, x_q, r_q, atom_name, res_name, res_num = import_charges_from_pqr(mesh_pqr_path)

    return q, x_q, r_q, atom_name, res_name, res_num


def read_tinker_radius(filename, radius_keyword="solute", solute_radius_type="PB"):
    r"""Reads atomic coordinates and maps the corresponding molecular radii from Tinker files.

    Parses the companion `.xyz` file for coordinates and atom types, and the `.key`
    file (or its linked parameter file) to extract and scale Van der Waals or optimized
    implicit solvent radii.

    Args:
        filename (str): Base filename without extension (used to look for `.xyz` and `.key` files).
        radius_keyword (str, optional): Type of radius set to extract. Choose between
            'solute' (optimized for implicit solvent calculations) or 'vdw'. Defaults to 'solute'.
        solute_radius_type (str, optional): Optimization tool variant for the solute keyword.
            Choose between 'PB', 'DDCOSMO', or 'GK'. Defaults to 'PB'.

    Returns:
        numpy.ndarray: 1-dim array of shape $(N,)$ containing the mapped atomic radii for
            each of the $N$ atoms in the structure.
    """

    file_xyz = filename + ".xyz"
    file_key = filename + ".key"

    with open(file_xyz, "r") as f:
        N = int(f.readline().split()[0])

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    r = np.zeros(N)

    atom_type = np.chararray(N, itemsize=10)

    i = 0
    header = 0

    xyz_file = open(file_xyz, "r")

    for line in xyz_file:
        line = line.split()

        if header == 1:

            x[i] = np.float64(line[2])
            y[i] = np.float64(line[3])
            z[i] = np.float64(line[4])

            atom_type[i] = line[5]

            i += 1

        header = 1

    xyz_file.close()

    atom_class = {}
    vdw_radii = {}
    solute_radii_PB = {}
    solute_radii_DD = {}
    solute_radii_GK = {}

    with open(file_key, "r") as f:

        line = f.readline().split()

        if line[0] == "parameters":

            file_key = line[1]

        print("Reading parameters from " + file_key)

    try:
        key_file = open(file_key, "r")
    except OSError:
        raise ValueError("Cannot find file " + file_key)
        return

    for line in key_file:

        line = line.split()

        if len(line) > 0:

            if line[0].lower() == "atom":

                atom_class[line[1]] = line[2]

            if line[0].lower() == "vdw" and line[1] not in vdw_radii:

                vdw_radii[line[1]] = np.float64(line[2]) / 2.0

            if line[0].lower() == "solute" and line[1] not in solute_radii_PB:

                solute_radii_PB[line[1]] = np.float64(line[2]) / 2.0
                solute_radii_DD[line[1]] = np.float64(line[3]) / 2.0
                solute_radii_GK[line[1]] = np.float64(line[4]) / 2.0

    key_file.close()

    for i in range(N):

        if radius_keyword == "vdw":
            r[i] = vdw_radii[atom_class[atom_type[i].decode()]]
        elif radius_keyword == "solute":
            if solute_radius_type == "PB":
                r[i] = solute_radii_PB[atom_type[i].decode()]
            elif solute_radius_type == "DDCOSMO":
                r[i] = solute_radii_DD[atom_type[i].decode()]
            elif solute_radius_type == "GK":
                r[i] = solute_radii_GK[atom_type[i].decode()]
            else:
                print(
                    "Unrecognized solute type radius. Choose between PB, DDCOSMO, and GK."
                )
                return
        else:
            print("Unrecognized keyword for radius. Choose between vdw and solute.")
            return

    return r


def find_multipole(multipole_list, connections, atom_type, pos, i, N):
    """Identifies the correct local multipole parameters for a specific atom based on its chemical environment.

    Filters available multipole definitions by matching the central atom type, followed
    by hierarchical filtering of the neighbors that define the local $z$-axis (must be bonded)
    and $x$-axis (the closest matching atom type in space).

    Args:
        multipole_list (list of list): Database of available multipole definitions. Each entry
            is a list/tuple containing at least `[central_type, zaxis_type, xaxis_type, ...]`.
        connections (dict or list of list): Adjacency list mapping each atom index to a
            list of its bonded neighbor indices.
        atom_type (numpy.ndarray): 1-dim array containing the string identifiers or types
            for all atoms in the system.
        pos (numpy.ndarray): 2-dim array of shape $(N, 3)$ containing the Cartesian coordinates
            of all atoms.
        i (int): Index of the target atom for which the multipole is being resolved.
        N (int): Total number of atoms in the molecular system.

    Returns:
        list: The matching multipole parameter entry from `multipole_list` that best fits
            the local frame rules.
    """
    #   filter possible multipoles by atom type
    atom_possible = []
    for j in range(len(multipole_list)):
        if atom_type[i] == multipole_list[j][0]:
            atom_possible.append(multipole_list[j])

    #   filter possible multipoles by z axis defining atom (needs to be bonded)
    #   only is atom_possible has more than 1 alternative
    if len(atom_possible) > 1:
        zaxis_possible = []
        for j in range(len(atom_possible)):
            for k in connections[i]:
                neigh_type = atom_type[k]
                if neigh_type == atom_possible[j][1]:
                    zaxis_possible.append(atom_possible[j])

        #       filter possible multipoles by x axis defining atom (no need to be bonded)
        #       only if zaxis_possible has more than 1 alternative
        if len(zaxis_possible) > 1:
            neigh_type = []
            for j in range(len(zaxis_possible)):
                neigh_type.append(zaxis_possible[j][2])

            xaxis_possible_atom = []
            for j in range(N):
                if atom_type[j] in neigh_type and i != j:
                    xaxis_possible_atom.append(j)

            dist = np.linalg.norm(pos[i, :] - pos[xaxis_possible_atom, :], axis=1)

            xaxis_at_index = np.where(np.abs(dist - np.min(dist)) < 1e-12)[0][0]
            xaxis_at = xaxis_possible_atom[xaxis_at_index]

            #           just check if it's not a connection
            if xaxis_at not in connections[i]:
                #                print 'For atom %i+1, x axis define atom is %i+1, which is not bonded'%(i,xaxis_at)
                for jj in connections[i]:
                    if jj in xaxis_possible_atom:
                        print(
                            "For atom %i+1, there was a bonded connnection available for x axis, but was not used"
                            % (i)
                        )

            xaxis_type = atom_type[xaxis_at]

            xaxis_possible = []
            for j in range(len(zaxis_possible)):
                if xaxis_type == zaxis_possible[j][2]:
                    xaxis_possible.append(zaxis_possible[j])

            if len(xaxis_possible) == 0:
                print("For atom %i+1 there is no possible multipole" % i)
            if len(xaxis_possible) > 1:
                print(
                    "For atom %i+1 there is more than 1 possible multipole, use last one"
                    % i
                )

        else:
            xaxis_possible = zaxis_possible

    else:
        xaxis_possible = atom_possible

    multipole = xaxis_possible[-1]

    return multipole


def load_tinker_multipoles_to_solute(solute):
    r"""Loads and resolves permanent multipoles, polarizabilities, and connectivity maps from Tinker files.

    Parses the molecular companion `.xyz` and `.key` files (or external parameter files)
    to extract coordinates, topology, permanent multipoles (monopoles, dipoles, and quadrupoles),
    and polarization groups. It computes local coordinate frames ($\mathbf{i}, \mathbf{j}, \mathbf{k}$)
    to rotate local multipole moments into the global reference frame and applies appropriate
    Bohr radius scalings.

    Args:
        solute (Solute): An instance of the Solute class containing configuration paths
            such as `xyz_path`, `radius_keyword`, and `solute_radius_type`.

    Returns:
        tuple: A tuple containing fifteen elements:
            - pos (numpy.ndarray): 2-dim array of shape $(N, 3)$ with the Cartesian positions of the multipoles.
            - q (numpy.ndarray): 1-dim array of shape $(N,)$ containing atomic charges (monopoles).
            - p (numpy.ndarray): 2-dim array of shape $(N, 3)$ containing global dipole moments.
            - Q (numpy.ndarray): 3-dim array of shape $(N, 3, 3)$ containing global quadrupole moments.
            - alpha (numpy.ndarray): 3-dim array of shape $(N, 3, 3)$ with atomic isotropic polarizability matrices.
            - r (numpy.ndarray): 1-dim array of shape $(N,)$ containing the mapped atomic radii.
            - mass (numpy.ndarray): 1-dim array of shape $(N,)$ containing atomic masses.
            - polar_group (numpy.ndarray): 1-dim array of shape $(N,)$ containing polarization group identifiers.
            - thole (numpy.ndarray): 1-dim array of shape $(N,)$ containing atomic Thole damping factors.
            - connections_12 (numpy.ndarray): 1-dim array containing flattened 1-2 (bonded) neighbor indices.
            - connections_13 (numpy.ndarray): 1-dim array containing flattened 1-3 neighbor indices.
            - pointer_connections_12 (numpy.ndarray): 1-dim index pointer array for parsing `connections_12`.
            - pointer_connections_13 (numpy.ndarray): 1-dim index pointer array for parsing `connections_13`.
            - p12scale (float): Polarization scaling factor for 1-2 interactions.
            - p13scale (float): Polarization scaling factor for 1-3 interactions.
    """
    filename = solute.xyz_path[:-4]  # remove extension

    file_xyz = filename + ".xyz"
    file_key = filename + ".key"

    with open(file_xyz, "r") as f:
        N = int(f.readline().split()[0])

    pos = np.zeros((N, 3))
    q = np.zeros(N)
    p = np.zeros((N, 3))
    Q = np.zeros((N, 3, 3))
    alpha = np.zeros((N, 3, 3))
    thole = np.zeros(N)
    mass = np.zeros(N)
    atom_type = np.chararray(N, itemsize=10)
    connections = np.empty(N, dtype=object)
    polar_group = -np.ones(N, dtype=np.int32)
    N_connections = 0
    header = 0

    file = open(file_xyz, "r").read().split("\n")
    for line in file:
        line = line.split()
        if header == 1 and len(line) > 0:
            atom_number = int(line[0]) - 1
            pos[atom_number, 0] = float(line[2])
            pos[atom_number, 1] = float(line[3])
            pos[atom_number, 2] = float(line[4])
            atom_type[atom_number] = line[5]
            connections[atom_number] = np.zeros(len(line) - 6, dtype=int)
            N_connections += len(line) - 6
            for i in range(6, len(line)):
                connections[atom_number][i - 6] = int(line[i]) - 1

        header = 1

    atom_class = {}
    atom_mass = {}
    polarizability = {}
    thole_factor = {}
    charge = {}
    dipole = {}
    quadrupole = {}
    polar_group_list = {}
    multipole_list = []
    multipole_flag = 0

    with open(file_key, "r") as f:
        line = f.readline().split()
        if line[0] == "parameters":
            file_key = line[1]

    try:
        file_k = open(file_key, "r").read().split("\n")
        print("Reading parameters from " + file_key)
    except OSError:
        raise ValueError(
            "Cannot find a key/prm file. Please use same file name as .xyz file, or check the path to the key file in your prm file."
        )
        return

    for line in file_k:
        line = line.split()

        if len(line) > 0:
            if line[0].lower() == "atom":
                atom_class[line[1]] = line[2]
                atom_mass[line[1]] = float(line[-2])

            if line[0].lower() == "polarize":
                polarizability[line[1]] = float(line[2])
                thole_factor[line[1]] = float(line[3])
                polar_group_list[line[1]] = np.chararray(len(line) - 4, itemsize=10)
                polar_group_list[line[1]][:] = line[4:]

            if line[0].lower() == "mpole-12-scale":
                pass
                # m12scale = float(line[1])
            if line[0].lower() == "mpole-13-scale":
                pass
                # m13scale = float(line[1])
            if line[0].lower() == "mpole-14-scale":
                pass
                # m14scale = float(line[1])
            if line[0].lower() == "mpole-15-scale":
                pass
                # m15scale = float(line[1])
            if line[0].lower() == "polar-12-scale":
                p12scale = float(line[1])
            if line[0].lower() == "polar-13-scale":
                p13scale = float(line[1])
            if line[0].lower() == "polar-14-scale":
                pass
                # p14scale = float(line[1])
            if line[0].lower() == "polar-15-scale":
                pass
                # p15scale = float(line[1])

            if line[0].lower() == "multipole" or (
                multipole_flag > 0 and multipole_flag < 5
            ):

                if multipole_flag == 0:
                    key = line[1]
                    z_axis = line[2]
                    x_axis = line[3]

                    if len(line) < 5:
                        x_axis = "0"

                    if len(line) > 5:
                        y_axis = line[4]
                    else:
                        y_axis = "0"

                    axis_type = "z_then_x"
                    if float(z_axis) == 0:
                        axis_type = "None"
                    if float(z_axis) != 0 and float(x_axis) == 0:
                        axis_type = "z_only"
                    if float(z_axis) < 0 or float(x_axis) < 0:
                        axis_type = "bisector"
                    if float(x_axis) < 0 and float(y_axis) < 0:  # not implemented yet
                        axis_type = "z_bisect"
                    if (
                        float(z_axis) < 0 and float(x_axis) < 0 and float(y_axis) < 0
                    ):  # not implemented yet
                        axis_type = "3_fold"

                    # Remove negative defining atom types
                    if z_axis[0] == "-":
                        z_axis = z_axis[1:]
                    if x_axis[0] == "-":
                        x_axis = x_axis[1:]
                    if y_axis[0] == "-":
                        y_axis = y_axis[1:]

                    multipole_list.append((key, z_axis, x_axis, y_axis, axis_type))

                    charge[(key, z_axis, x_axis, y_axis, axis_type)] = float(line[-1])
                if multipole_flag == 1:
                    dipole[(key, z_axis, x_axis, y_axis, axis_type)] = np.array(
                        [float(line[0]), float(line[1]), float(line[2])]
                    )
                if multipole_flag == 2:
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)] = np.zeros(
                        (3, 3)
                    )
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][0, 0] = float(
                        line[0]
                    )
                if multipole_flag == 3:
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][1, 0] = float(
                        line[0]
                    )
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][0, 1] = float(
                        line[0]
                    )
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][1, 1] = float(
                        line[1]
                    )
                if multipole_flag == 4:
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][2, 0] = float(
                        line[0]
                    )
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][0, 2] = float(
                        line[0]
                    )
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][2, 1] = float(
                        line[1]
                    )
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][1, 2] = float(
                        line[1]
                    )
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][2, 2] = float(
                        line[2]
                    )
                    multipole_flag = -1

                multipole_flag += 1

    polar_group_counter = 0
    for i in range(N):
        #       Get polarizability
        alpha[i, :, :] = np.identity(3) * polarizability[atom_type[i].decode()]

        #       Get Thole factor
        thole[i] = thole_factor[atom_type[i].decode()]

        #       Get mass
        mass[i] = atom_mass[atom_type[i].decode()]

        #       Find atom polarization group
        if polar_group[i] == -1:
            #           Check with connections if there is a group member already assigned
            for j in connections[i]:
                if (
                    atom_type[j] in polar_group_list[atom_type[i].decode()][:]
                    and polar_group[j] != -1
                ):
                    if polar_group[i] == -1:
                        polar_group[i] = polar_group[j]
                    elif polar_group[i] != polar_group[j]:
                        print("double polarization group assigment here!")
            #           if no other group members are found, create a new group
            if polar_group[i] == -1:
                polar_group[i] = np.int32(polar_group_counter)
                polar_group_counter += 1

            #           Now, assign group number to connections in the same group
            for j in connections[i]:
                if atom_type[j] in polar_group_list[atom_type[i].decode()][:]:
                    if polar_group[j] == -1:
                        polar_group[j] = polar_group[i]
                    elif polar_group[j] != polar_group[i]:
                        print("double polarization group assigment here too!")

        multipole = find_multipole(
            multipole_list, connections, atom_type.decode(), pos, i, N
        )

        #       Find local axis
        #       Find z defining atom (needs to be bonded)
        z_atom = -1
        for k in connections[i]:
            neigh_type = atom_type[k].decode()
            if neigh_type == multipole[1]:
                if z_atom == -1:
                    z_atom = k

        #       Find x defining atom (no need to be bonded)
        #       First, look within 1-2 bonded atoms
        x_atom = -1

        for k in connections[i]:
            neigh_type = atom_type[k].decode()
            if neigh_type == multipole[2] and k != z_atom:
                if x_atom == -1:
                    x_atom = k

        #       Next, look within 1-3 bonded atoms
        if x_atom == -1:
            for k in connections[i]:
                for ll in connections[k]:
                    neigh_type = atom_type[ll].decode()
                    if neigh_type == multipole[2] and ll != i and ll != z_atom:
                        if x_atom == -1:
                            x_atom = ll

        #       Else, look within nonbonded atoms
        if x_atom == -1:
            neigh_type = multipole[2]
            x_possible_atom = []
            for j in range(N):
                if atom_type[j] == neigh_type and i != j and j != z_atom:
                    x_possible_atom.append(j)

            if len(x_possible_atom) > 0:
                dist = np.linalg.norm(pos[i, :] - pos[x_possible_atom, :], axis=1)

                x_atom_index = np.where(np.abs(dist - np.min(dist)) < 1e-12)[0][0]
                x_atom = x_possible_atom[x_atom_index]

        if x_atom == -1 and multipole[4] == "z_only":  # no need for an x_atom
            x_atom = -2

        if z_atom == -1 or x_atom == -1:  # for example, in the sphere case
            i_local = np.array([1, 0, 0])
            j_local = np.array([0, 1, 0])
            k_local = np.array([0, 0, 1])

        else:
            r12 = pos[z_atom, :] - pos[i, :]
            r13 = pos[x_atom, :] - pos[i, :]
            if multipole[4] == "z_then_x":
                k_local = r12 / np.linalg.norm(r12)
                i_local = (r13 - np.dot(r13, k_local) * k_local) / np.linalg.norm(
                    r13 - np.dot(r13, k_local) * k_local
                )
                j_local = np.cross(k_local, i_local)

            elif multipole[4] == "bisector":
                k_local = r12 / np.linalg.norm(r12) + r13 / np.linalg.norm(r13)
                k_local = k_local / np.linalg.norm(k_local)
                i_local = (r13 - np.dot(r13, k_local) * k_local) / np.linalg.norm(
                    r13 - np.dot(r13, k_local) * k_local
                )
                j_local = np.cross(k_local, i_local)

            elif multipole[4] == "z_only":
                k_local = r12 / np.linalg.norm(r12)

                dX = np.array([1.0, 0.0, 0.0])
                dot = k_local[0]
                if abs(dot) > 0.866:
                    dX[0] = 0.0
                    dX[1] = 1.0
                    dot = k_local[1]

                dX -= dot * k_local
                i_local = dX / np.linalg.norm(dX)

                j_local = np.cross(k_local, i_local)

        #       Assign charge
        q[i] = charge[multipole]

        #       Find rotation matrix
        A = np.identity(3)
        A[:, 0] = i_local
        A[:, 1] = j_local
        A[:, 2] = k_local

        bohr = 0.52917721067
        #       Assign dipole
        p[i, :] = np.dot(A, dipole[multipole]) * bohr

        #       Assign quadrupole
        for ii in range(3):
            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        Q[i, ii, j] += (
                            A[ii, k]
                            * A[j, m]
                            * quadrupole[multipole][k, m]
                            * bohr**2
                            * 2
                        )  # x2 to agree with Tinker's formulation (they include 1/2 in Q)

    #   Connections list
    #   1-2 connections (already computed, just put into 1D array)
    connections_12 = np.zeros(N_connections, dtype=np.int32)
    pointer_connections_12 = np.zeros(
        N + 1, dtype=np.int32
    )  # pointer to beginning of interaction list
    for i in range(N):
        pointer_connections_12[i + 1] = pointer_connections_12[i] + len(connections[i])
        start = pointer_connections_12[i]
        stop = pointer_connections_12[i + 1]
        connections_12[start:stop] = connections[i]

    if N < 2:  # if no 1-2 connections
        connections_12 = np.zeros(N)  # this avoids a GPU error later

    #   1-3 connections
    connections_13 = np.zeros(int(N_connections * N_connections / N), dtype=np.int32)
    pointer_connections_13 = np.zeros(
        N + 1, dtype=np.int32
    )  # pointer to beginning of interaction list

    if N > 2:  # ions and diatomic molecules have no 1-3 connections
        for i in range(N):
            possible_connections = np.concatenate(connections[connections[i]])
            possible_connections = np.unique(
                possible_connections
            )  # filter out repeated connections
            index_self = np.where(possible_connections == i)[0]  # remove self atom
            possible_connections = np.delete(possible_connections, index_self)
            pointer_connections_13[i + 1] = pointer_connections_13[i] + len(
                possible_connections
            )

            start = pointer_connections_13[i]
            end = pointer_connections_13[i + 1]
            connections_13[start:end] = possible_connections

        connections_13 = connections_13[: pointer_connections_13[-1]]
    else:
        connections_13 = np.zeros(N)  # this avoids a GPU error later

    r = read_tinker_radius(
        filename,
        radius_keyword=solute.radius_keyword,
        solute_radius_type=solute.solute_radius_type,
    )

    return (
        pos,
        q,
        p,
        Q,
        alpha,
        r,
        mass,
        polar_group,
        thole,
        connections_12,
        connections_13,
        pointer_connections_12,
        pointer_connections_13,
        p12scale,
        p13scale,
    )


def generate_msms_mesh_import_tinker_multipoles(solute):
    """Generates the molecular surface mesh and extracts comprehensive Tinker multipole parameters.

    Handles the generation of `.xyzr` files, orchestrates the execution of the
    specified surface mesh generator (MSMS or NanoShaper) to build the molecular boundary,
    and unpacks all electrostatic, polarization, and connectivity matrices required for
    implicit solvent modeling.

    Args:
        solute (Solute): An instance of the Solute class containing mesh options (generator,
            density, probe radius) and Tinker file paths.

    Returns:
        tuple: A tuple containing sixteen elements:
            - grid (bempp.api.Grid): The generated/imported Bempp Grid surface object.
            - x_q (numpy.ndarray): 2-dim array of shape $(N, 3)$ with the Cartesian positions of the multipoles.
            - q (numpy.ndarray): 1-dim array of shape $(N,)$ containing atomic charges (monopoles).
            - d (numpy.ndarray): 2-dim array of shape $(N, 3)$ containing global dipole moments.
            - Q (numpy.ndarray): 3-dim array of shape $(N, 3, 3)$ containing global quadrupole moments.
            - alpha (numpy.ndarray): 3-dim array of shape $(N, 3, 3)$ with atomic isotropic polarizability matrices.
            - r_q (numpy.ndarray): 1-dim array of shape $(N,)$ containing the mapped atomic radii.
            - mass (numpy.ndarray): 1-dim array of shape $(N,)$ containing atomic masses.
            - polar_group (numpy.ndarray): 1-dim array of shape $(N,)$ containing polarization group identifiers.
            - thole (numpy.ndarray): 1-dim array of shape $(N,)$ containing atomic Thole damping factors.
            - connections_12 (numpy.ndarray): 1-dim array containing flattened 1-2 (bonded) neighbor indices.
            - connections_13 (numpy.ndarray): 1-dim array containing flattened 1-3 neighbor indices.
            - pointer_connections_12 (numpy.ndarray): 1-dim index pointer array for parsing `connections_12`.
            - pointer_connections_13 (numpy.ndarray): 1-dim index pointer array for parsing `connections_13`.
            - p12scale (float): Polarization scaling factor for 1-2 interactions.
            - p13scale (float): Polarization scaling factor for 1-3 interactions.
    """

    mesh_dir = os.path.abspath("mesh_temp/")
    if solute.save_mesh_build_files:
        mesh_dir = solute.mesh_build_files_dir

    if not os.path.exists(mesh_dir):
        try:
            os.mkdir(mesh_dir)
        except OSError:
            print("Creation of the directory %s failed" % mesh_dir)

    (
        x_q,
        q,
        d,
        Q,
        alpha,
        r_q,
        mass,
        polar_group,
        thole,
        connections_12,
        connections_13,
        pointer_connections_12,
        pointer_connections_13,
        p12scale,
        p13scale,
    ) = load_tinker_multipoles_to_solute(solute)

    mesh_xyzr_path = os.path.join(mesh_dir, solute.solute_name + ".xyzr")

    # Generate xyzr for mesh
    N = len(q)
    data_xyzr = np.zeros((N, 4))
    data_xyzr[:, :3] = x_q[:, :]
    data_xyzr[:, 3] = r_q[:]
    np.savetxt(mesh_xyzr_path, data_xyzr, fmt="%1.4f")

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
            cavity_cutoff=solute.cavity_cutoff,
            fill_cavities=solute.fill_cavities,
        )

    mesh_off_path = os.path.join(mesh_dir, solute.solute_name + ".off")

    grid = import_msms_mesh(mesh_face_path, mesh_vert_path)

    grid = check_cavity(
        grid, fill_cavities=solute.fill_cavities, volume_cutoff=solute.cavity_cutoff
    )
    if solute.mesh_generator == "msms":
        grid = fix_mesh(grid)

    if solute.save_mesh_build_files:
        solute.mesh_xyzr_path = mesh_xyzr_path
        solute.mesh_face_path = mesh_face_path
        solute.mesh_vert_path = mesh_vert_path
        solute.mesh_off_path = mesh_off_path
    else:
        shutil.rmtree(mesh_dir)

    return (
        grid,
        x_q,
        q,
        d,
        Q,
        alpha,
        r_q,
        mass,
        polar_group,
        thole,
        connections_12,
        connections_13,
        pointer_connections_12,
        pointer_connections_13,
        p12scale,
        p13scale,
    )


def convert_units(units, magnitude="potential", temperature=298.15):
    r"""Compute the scalar conversion factor from atomic electrostatic units to a target unit system.

    The input unit string is normalized and matched against the supported aliases for
    electrostatic potentials, derivatives, energies, and forces. The function then returns
    the corresponding conversion factor and a human-readable label.

    Args:
        units (str/object): Target unit identifier. Supported aliases include `mv`, `mv_a`,
            `mv_m`, `v`, `volt`, `volts`, `v_m`, `v_a`, `kt_e`, `kt_a`, `kt`,
            `kj_mol_e`, `kj_mol`, `kj_mola`, `kj_mol_a`, `kcal_mol_e`, `kcal_mol`,
            `kcal_mola`, `kcal_mol_a`, `e_eps0_angs`, `e_eps0_ang`, `atomic`, and `pn`.
            The `pn` alias is only valid when `magnitude='force'`.
        magnitude (str, optional): Physical quantity being converted. Must be one of
            `potential`, `d_potential`, `energy`, or `force`. Defaults to `potential`.
        temperature (float, optional): Absolute temperature in Kelvin used for thermal energy
            conversions. Defaults to 298.15.

    Returns:
        tuple: A tuple containing the conversion factor and a descriptive unit label.

    Raises:
        ValueError: If `pn` is requested for a magnitude other than `force`, or if the
            requested magnitude is not recognized.
    """
    units = str(units).strip().lower().replace("-", "_").replace(" ", "_")
    units = units.replace("__", "_")
    magnitude = str(magnitude).strip().lower()

    qe = 1.60217663e-19
    eps0 = 8.8541878128e-12
    ang_to_m = 1e-10
    kb = 1.380649e-23
    kT = kb * temperature
    Na = 6.02214076e23

    to_V = qe / (eps0 * ang_to_m)

    if units in ["mv_m", "mv_a", "mv"]:
        factor_base, label_base = to_V * 1000, "mV"
    elif units in ["v", "volt", "volts", "v_m", "v_a"]:
        factor_base, label_base = to_V, "V"
    elif units in ["kt_e", "kt_a", "kt"]:
        factor_base, label_base = to_V / (kT / qe), "kT/e"
    elif units in ["kj_mol_e", "kj_mol", "kj_mola", "kj_mol_a"]:
        factor_base, label_base = to_V * (qe * Na / 1000), "kJ/mol"
    elif units in ["kcal_mol_e", "kcal_mol", "kcal_mola", "kcal_mol_a"]:
        factor_base, label_base = to_V * (qe * Na / (4.184 * 1000)), "kcal/mol"
    elif units in ["e_eps0_angs", "e_eps0_ang", "atomic"]:
        factor_base, label_base = 1.0, "e/(eps0*A)"
    elif units in ["pn"]:
        if magnitude != "force":
            raise ValueError(
                f"Unit 'pN' is only valid for magnitude='force', not '{magnitude}'."
            )
        factor_base, label_base = (qe**2 / (eps0 * ang_to_m**2)) / 1e-12, "pN"
        return factor_base, label_base
    else:
        if magnitude in ["potential", "d_potential"]:
            print(
                f"Warning: Unit '{units}' not recognized for {magnitude}. Defaulting to mV."
            )
            factor_base, label_base = to_V * 1000, "mV"
        else:
            print(
                f"Warning: Unit '{units}' not recognized for {magnitude}. Defaulting to kcal/mol."
            )
            factor_base, label_base = to_V * (qe * Na / (4.184 * 1000)), "kcal/mol"

    if magnitude == "potential":
        if "mol" in label_base:
            label_base += "/e"
        return factor_base, label_base

    elif magnitude in ["d_potential"]:
        if units in ["e_eps0_angs", "e_eps0_ang", "atomic"]:
            return 1.0, "e/(eps0*A**2)"
        elif units in ["v_m"]:
            return to_V / ang_to_m, "V/m"
        elif units in ["mv_m"]:
            return (to_V * 1000) / ang_to_m, "mV/m"
        return factor_base, f"{label_base}/A"

    elif magnitude == "energy":
        if "mol" in label_base:
            return factor_base, label_base
        elif label_base == "kT/e":
            return factor_base, "kT"
        elif label_base == "e/(eps0*A)":
            return 1.0, "e**2/(eps0*A)"
        else:
            return factor_base, f"{label_base}*e"

    elif magnitude == "force":
        if "mol" in label_base:
            return factor_base, f"{label_base}/A"
        elif label_base == "kT/e":
            return factor_base, "kT/A"
        elif label_base == "e/(eps0*A)":
            return 1.0, "e**2/(eps0*A**2)"
        else:
            return factor_base, f"{label_base}*e/A"

    else:
        raise ValueError(
            f"Magnitude '{magnitude}' not recognized. Choose from: potential, d_potential, energy, force."
        )
