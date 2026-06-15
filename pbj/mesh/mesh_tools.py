import trimesh
import numpy as np
import subprocess
import os
import bempp_cl as bempp
import bempp_cl.api
import platform
import shutil


def check_cavity(mesh, fill_cavities=True, volume_cutoff=11.4):
    r"""Detects, filters, and removes internal or isolated cavities within a mesh.

        Splits a disconnected mesh into its separate connected components, treating the
        largest component as the main body. The remaining components (cavities) are 
        evaluated and removed if they reside outside the main body or fall below a 
        specified volume threshold.

        Args:
            mesh (object): The input mesh object containing `.vertices.T` and 
                `.elements.T` attributes compatible with trimesh initialization.
            fill_cavities (bool, optional): If True, proceeds with cavity detection 
                and filtering. If False, skips processing and returns the original mesh. 
                Defaults to True.
            volume_cutoff (float, optional): The volume threshold below which smaller 
                internal cavities will be flagged for removal. Defaults to 11.4.

        Returns:
            bempp_cl.api.Grid or object: A new BEMPP Grid object generated from the 
                cleaned largest mesh component, or the original input mesh if no 
                cavities were processed.
    """
    mesh_raw = trimesh.Trimesh(vertices=mesh.vertices.T, faces=mesh.elements.T)
    mesh_split = mesh_raw.split()
    if len(mesh_split) == 1 or not fill_cavities:
        print("No cavities detected in the mesh")
        return mesh

    largest_mesh = max(mesh_split, key=lambda m: m.volume)
    idx_remove = []
    for i in range(len(mesh_split)):  # remove mesh cavities off the largest one
        if not any(
            largest_mesh.contains(mesh_split[i].vertices[0:1, :])
        ):  # evaluate one point to discard
            idx_remove.append(i)
            print(
                "Cavity far off the largest mesh detected and removed with volume {:.2f}.".format(
                    mesh_split[i].volume
                )
            )
        if abs(mesh_split[i].volume) > volume_cutoff:
            idx_remove.append(i)
            print(
                "Small inner cavity detected and removed with volume {:.2f}.".format(
                    mesh_split[i].volume
                )
            )
    mesh_split = [mesh_split[i] for i in range(len(mesh_split)) if i not in idx_remove]
    print("{} cavities detected and removed.".format(len(idx_remove)))

    return bempp_cl.api.Grid(largest_mesh.vertices.T, largest_mesh.faces.T)


def fix_mesh(mesh):
    r"""Loads a mesh from text files and iteratively attempts to repair it into a watertight surface.

    This function reads face indices and vertex coordinates from separate text files,
    initializes a `trimesh` object, and applies initial healing operations. If the mesh
    is not watertight, it runs an iterative loop to identify broken faces and snap/merge
    vertices that lie within a small distance tolerance.

    Args:
        mesh_face_path (str): File path to the text file containing the face indices.
        mesh_vert_path (str): File path to the text file containing the vertex
            coordinates (X, Y, Z).

    Returns:
        trimesh.Trimesh: The repaired and processed mesh object.

    Notes:
        Prints a warning to the console if the mesh cannot be made completely
        watertight within the maximum iteration limit (20).
    """
    mesh = trimesh.Trimesh(vertices=mesh.vertices.T, faces=mesh.elements.T)
    mesh.fill_holes()
    mesh.process()
    iter_limit = 20
    iteration = 0
    while not mesh.is_watertight and iteration < iter_limit:
        merge_tolerance = 0.05
        needy_faces = trimesh.repair.broken_faces(mesh)
        for vert_nf in mesh.faces[needy_faces]:
            for nf in vert_nf:
                for c, check in enumerate(
                    np.linalg.norm(mesh.vertices[vert_nf] - mesh.vertices[nf], axis=1)
                ):
                    if (check < merge_tolerance) & (0 < check):
                        mesh.vertices[nf] = mesh.vertices[vert_nf[c]]
        iteration += 1
    if iteration > iter_limit - 1:
        print("Warning: Mesh is not watertight")
    mesh.fill_holes()
    mesh.process()
    return bempp_cl.api.Grid(mesh.vertices.T, mesh.faces.T)


# Revisar función, elegir paquete correcto o buscar opción de ejecutable:
def convert_pdb2pqr(mesh_pdb_path, mesh_pqr_path, force_field, str_flag=""):
    r"""Invokes the PDB2PQR tool via subprocess to parameterize a PDB structure into a PQR file.

    Assigns atomic charges and radii based on the specified force field, generating
    the topology configuration required for downstream continuum electrostatics.

    Args:
        mesh_pdb_path (str): Absolute file path to the source `.pdb` file.
        mesh_pqr_path (str): Absolute target path for the output parameterized `.pqr` file.
        force_field (str): The capitalization-agnostic force field identifier
            (e.g., 'AMBER', 'CHARMM', 'PARSE', 'TYL06').
        str_flag (str, optional): Additional command-line flags to pass directly to the
            `pdb2pqr30` executable. Defaults to an empty string.

    Returns:
        None
    """
    force_field = force_field.upper()
    if str_flag:
        subprocess.call(
            ["pdb2pqr30", str_flag, "--ff=" + force_field, mesh_pdb_path, mesh_pqr_path]
        )
    else:
        subprocess.call(
            ["pdb2pqr30", "--ff=" + force_field, mesh_pdb_path, mesh_pqr_path]
        )


# Funciona bien:
def convert_pqr2xyzr(mesh_pqr_path, mesh_xyzr_path):
    """Parses a PQR file and extracts coordinates and radii into a simplified XYZR format.

    Filters for 'ATOM' records and writes out rows containing only the Cartesian
    coordinates ($x, y, z$) and the atomic radius ($r$) for each atom.

    Args:
        mesh_pqr_path (str): Absolute file path to the input `.pqr` file.
        mesh_xyzr_path (str): Absolute target path for the output `.xyzr` file.

    Returns:
        None
    """
    pqr_file = open(mesh_pqr_path, "r")
    pqr_data = pqr_file.read().split("\n")
    xyzr_file = open(mesh_xyzr_path, "w")
    for line in pqr_data:
        line = line.split()
        if len(line) == 0 or line[0] != "ATOM":
            continue
        xyzr_file.write(
            line[5] + "\t" + line[6] + "\t" + line[7] + "\t" + line[9] + "\n"
        )
    pqr_file.close()
    xyzr_file.close()


# Probar en Linux:
def generate_msms_mesh(mesh_xyzr_path, output_dir, output_name, density, probe_radius):
    """Generates a Solvent-Excluded Surface (SES) mesh using the external MSMS executable.

    Produces paired `.face` and `.vert` files without headers in the specified target directory.

    Args:
        mesh_xyzr_path (str): Absolute file path to the input `.xyzr` structural file.
        output_dir (str): Absolute path to the directory where the output mesh files will be stored.
        output_name (str): Base filename string for the generated `.face` and `.vert` files.
        density (float): Triangle density on the molecular surface (typically $1.0$ for large structures,
            $3.0$ for smaller systems).
        probe_radius (float): Radius of the rolling solvent probe (typically $1.4\text{ Å}$ for water).

    Returns:
        None
    """
    from pbj import PBJ_PATH

    path = os.path.join(output_dir, output_name)
    msms_dir = os.path.join(PBJ_PATH, "mesh", "ExternalSoftware", "MSMS", "")
    if platform.system() == "Linux":
        external_file = "msms"
        os.system("chmod +x " + msms_dir + external_file)
    elif platform.system() == "Windows":
        external_file = "msms.exe"
    command = (
        msms_dir
        + external_file
        + " -if "
        + mesh_xyzr_path
        + " -of "
        + path
        + " -p "
        + str(probe_radius)
        + " -d "
        + str(density)
        + " -no_header"
    )
    print(command)
    os.system(command)


# Averiguar por qué terminamos en dos directorios más arriba
def generate_nanoshaper_mesh(
    mesh_xyzr_path,
    output_dir,
    output_name,
    density,
    probe_radius,
    save_mesh_build_files,
    cavity_cutoff=11.4,
    fill_cavities=True,
):
    """Generates a molecular surface mesh using NanoShaper via a temporary workspace.

    Dynamically populates a `surfaceConfiguration.prm` parameter template, switches
    working directories to execute the correct architecture-dependent binary, and
    cleans up intermediate files depending on the persistence configuration.

    Args:
        mesh_xyzr_path (str): Absolute file path to the source `.xyzr` file.
        output_dir (str): Absolute path to the destination directory for the final mesh.
        output_name (str): Base filename prefix for the generated `.face` and `.vert` files.
        density (float): Grid scale resolution value passed directly to NanoShaper.
        probe_radius (float): Rolling probe sphere radius used to construct the analytical interface.
        save_mesh_build_files (bool): If True, retains the raw NanoShaper working directory
            (`/nanotemp`) instead of deleting it.
        cavity_cutoff (float): Cutoff value for cavity detection.
        fill_cavities (bool): If True, fills detected cavities.

    Returns:
        None
    """
    from pbj import PBJ_PATH

    nanoshaper_dir = os.path.join(
        PBJ_PATH, "mesh", "ExternalSoftware", "NanoShaper", ""
    )
    nanoshaper_temp_dir = os.path.join(output_dir, "nanotemp", "")

    if not os.path.exists(nanoshaper_temp_dir):
        os.makedirs(nanoshaper_temp_dir)

    # Execute NanoShaper
    config_template_file = open(nanoshaper_dir + "config", "r")
    config_file = open(nanoshaper_temp_dir + "surfaceConfiguration.prm", "w")
    for line in config_template_file:
        if "XYZR_FileName" in line:
            line = "XYZR_FileName = " + mesh_xyzr_path + " \n"
        elif "Grid_scale" in line:
            line = "Grid_scale = {:04.1f} \n".format(density)
        elif "Probe_Radius" in line:
            line = "Probe_Radius = {:03.1f} \n".format(probe_radius)
        elif "Conditional_Volume_Filling_Value" in line:
            line = "Conditional_Volume_Filling_Value = {:03.1f} \n".format(
                cavity_cutoff
            )
        elif "Cavity_Detection_Filling" in line:
            line = "Cavity_Detection_Filling = {:s} \n".format(
                str(fill_cavities).lower()
            )
        config_file.write(line)

    config_file.close()
    config_template_file.close()

    original_dir = os.getcwd()
    os.chdir(nanoshaper_temp_dir)
    if platform.system() == "Linux":
        os.system("chmod +x " + nanoshaper_dir + "NanoShaper")
        os.system(nanoshaper_dir + "NanoShaper")
    elif platform.system() == "Windows":
        if platform.architecture()[0] == "32bit":
            os.system(
                nanoshaper_dir
                + "NanoShaper32.exe"
                + " "
                + nanoshaper_temp_dir
                + "surfaceConfiguration.prm"
            )
        elif platform.architecture()[0] == "64bit":
            os.system(
                nanoshaper_dir
                + "NanoShaper64.exe"
                + " "
                + nanoshaper_temp_dir
                + "surfaceConfiguration.prm"
            )
    os.chdir("..")

    try:
        vert_file = open(nanoshaper_temp_dir + "triangulatedSurf.vert", "r")
        vert = vert_file.readlines()
        vert_file.close()
        face_file = open(nanoshaper_temp_dir + "triangulatedSurf.face", "r")
        face = face_file.readlines()
        face_file.close()

        vert_file = open(output_name + ".vert", "w")
        vert_file.write("".join(vert[3:]))
        vert_file.close()
        face_file = open(output_name + ".face", "w")
        face_file.write("".join(face[3:]))
        face_file.close()

        if not save_mesh_build_files:
            shutil.rmtree(nanoshaper_temp_dir)

    except (OSError, FileNotFoundError):
        print("The file doesn't exist or it wasn't created by NanoShaper")

    finally:
        os.chdir(original_dir)


def convert_msms2off(mesh_face_path, mesh_vert_path, mesh_off_path):
    """Converts a raw MSMS `.face` and `.vert` file pairing into a unified Geomview OFF mesh file.

    Applies a 1-based to 0-based index shift to the surface triangulation matrix during
    the reformatting.

    Args:
        mesh_face_path (str): Absolute file path to the input MSMS `.face` file.
        mesh_vert_path (str): Absolute file path to the input MSMS `.vert` file.
        mesh_off_path (str): Absolute target file path for the output `.off` file.

    Returns:
        None
    """
    face = open(mesh_face_path, "r").read()
    vert = open(mesh_vert_path, "r").read()

    faces = np.vstack(np.char.split(face.split("\n")[0:-1]))[:, :3].astype(int) - 1
    verts = np.vstack(np.char.split(vert.split("\n")[0:-1]))[:, :3].astype(float)

    data = open(mesh_off_path, "w")
    data.write("OFF" + "\n")
    data.write(str(verts.shape[0]) + " " + str(faces.shape[0]) + " " + str(0) + "\n")
    for vert in verts:
        data.write(str(vert[0]) + " " + str(vert[1]) + " " + str(vert[2]) + "\n")
    for face in faces:
        data.write(
            "3" + " " + str(face[0]) + " " + str(face[1]) + " " + str(face[2]) + "\n"
        )


def import_msms_mesh(mesh_face_path, mesh_vert_path):
    """Loads MSMS surface outputs directly into a Bempp Grid.

    Parses vertices and faces into independent NumPy arrays, transforms them into the
    required column-vector orientation, and instantiates the discrete Bempp mesh.

    Args:
        mesh_face_path (str): Absolute path to the source `.face` file.
        mesh_vert_path (str): Absolute path to the source `.vert` file.

    Returns:
        bempp_cl.api.Grid: The discrete boundary element surface grid object.
    """
    face = open(mesh_face_path, "r").read()
    vert = open(mesh_vert_path, "r").read()

    faces = np.vstack(np.char.split(face.split("\n")[0:-1]))[:, :3].astype(int) - 1
    verts = np.vstack(np.char.split(vert.split("\n")[0:-1]))[:, :3].astype(float)

    grid = bempp.api.Grid(verts.transpose(), faces.transpose())
    return grid


def import_off_mesh(mesh_off_path):
    """Loads a unified Geomview OFF file using Bempp's native I/O utilities.

    Args:
        mesh_off_path (str): Absolute path to the source `.off` file.

    Returns:
        bempp_cl.api.Grid: The instantiated discrete boundary element mesh.
    """
    grid = bempp.api.import_grid(mesh_off_path)
    return grid


def density_to_nanoshaper_grid_scale_conversion(mesh_density):
    r"""Converts a standard face triangle density value into NanoShaper's internal spatial grid scale.

    Applies the empirically fitted power-law relationship:
    $$s = \text{round}\left(0.797 \cdot d^{0.507}, 2\right)$$
    where $d$ is the targeted surface triangle density and $s$ is the grid scale.

    Args:
        mesh_density (float): Targeted triangle surface density.

    Returns:
        float: The rounded grid scale parameter for the NanoShaper initialization file.
    """
    grid_scale = round(
        0.797 * (mesh_density**0.507), 2
    )  # Emperical relation found by creating meshes using nanoshaper and calculating their density
    return grid_scale
