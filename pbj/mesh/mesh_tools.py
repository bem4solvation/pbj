import time
import trimesh
import numpy as np
import subprocess
import os
import bempp.api
import platform
import shutil

# from pbj import PBJ_PATH # circular import error
#Preguntar por qué se encadenan estas funciones:
PBJ_PATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def fix_mesh(mesh):

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
        print(" not watertight")

    mesh.fill_holes()
    mesh.process()

    return mesh

#Revisar función, elegir paquete correcto o buscar opción de ejecutable:
def convert_pdb2pqr(mesh_pdb_path, mesh_pqr_path, force_field, str_flag=""):
    """
    Using pdb2pqr from APBS transform a pdb file into a pqr
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

#Funciona bien:
def convert_pqr2xyzr(mesh_pqr_path, mesh_xyzr_path):
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

#Probar en Linux:
def generate_msms_mesh(mesh_xyzr_path, output_dir, output_name, density, probe_radius):
    path = os.path.join(output_dir, output_name)
    msms_dir = os.path.join(PBJ_PATH, 'ExternalSoftware','MSMS','')
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
    os.system(command)


#Averiguar por qué terminamos en dos directorios más arriba
def generate_nanoshaper_mesh(
    mesh_xyzr_path,
    output_dir,
    output_name,
    density,
    probe_radius,
    save_mesh_build_files,
):
    """
    Usar path absoluto de mesh_xyzr_path y de output_name
    """
    nanoshaper_dir = os.path.join(PBJ_PATH,'ExternalSoftware','Nanoshaper','')
    nanoshaper_temp_dir = os.path.join(output_dir, 'nanotemp','')
    #if platform.system() == "Linux":
    #    nanoshaper_dir = os.path.join(PBJ_PATH, "ExternalSoftware/NanoShaper/")
    #    nanoshaper_temp_dir = os.path.join(output_dir, "nanotemp/")
    #elif platform.system() == "Windows":
    #    nanoshaper_dir = os.path.join(PBJ_PATH, "ExternalSoftware\\NanoShaper\\")
    #    nanoshaper_temp_dir = os.path.join(output_dir, "nanotemp\\")

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

        config_file.write(line)

    config_file.close()
    config_template_file.close()

    #Probar en Linux
    os.chdir(nanoshaper_temp_dir)
    if platform.system() == "Linux":
        os.system(nanoshaper_dir + "NanoShaper")
        os.system("chmod +x " + nanoshaper_dir + "NanoShaper")
    elif platform.system() == "Windows":
        if platform.architecture()[0] == '32bit':
            os.system(nanoshaper_dir + "NanoShaper32.exe" + " " + nanoshaper_temp_dir+ 'surfaceConfiguration.prm')
        elif platform.architecture()[0] == '64bit':
            os.system(nanoshaper_dir + "NanoShaper64.exe" + " " + nanoshaper_temp_dir+ 'surfaceConfiguration.prm')
    os.chdir('..')
    
    try:
        vert_file = open(nanoshaper_temp_dir + 'triangulatedSurf.vert', "r")
        vert = vert_file.readlines()
        vert_file.close()
        face_file = open(nanoshaper_temp_dir + 'triangulatedSurf.face', "r")
        face = face_file.readlines()
        face_file.close()

        ##os.remove(output_name + ".vert")
        ##os.remove(output_name + ".face")

        vert_file = open(output_name + ".vert", "w")
        vert_file.write("".join(vert[3:]))
        vert_file.close()
        face_file = open(output_name + ".face", "w")
        face_file.write("".join(face[3:]))
        face_file.close()
        # Ya termina el crear los face y ver

        if not save_mesh_build_files:
            shutil.rmtree(nanoshaper_temp_dir)
            #shutil.copy(nanoshaper_temp_dir + 'triangulatedSurf.face', output_name + ".face")
            #shutil.copy(nanoshaper_temp_dir + 'triangulatedSurf.vert', output_name + ".vert")
        #files = os.listdir(nanoshaper_temp_dir)
        #for file in files:
        #    os.remove(os.path.join(nanoshaper_temp_dir+)
        #os.system("rm -r " + nanoshaper_temp_dir)
        

        """
        vert_file = open(output_name + ".vert", "r")
        vert = vert_file.readlines()
        vert_file.close()
        face_file = open(output_name + ".face", "r")
        face = face_file.readlines()
        face_file.close()

        ##os.remove(output_name + ".vert")
        ##os.remove(output_name + ".face")

        vert_file = open(output_name + ".vert", "w")
        vert_file.write("".join(vert[3:]))
        vert_file.close()
        face_file = open(output_name + ".face", "w")
        face_file.write("".join(face[3:]))
        face_file.close()
        """
    except OSError:
        print('El archivo no existe o no fue creado por NanoShaper')
    except FileExistsError:
        pass
    # Vestigios de la version pasada
    """
    if platform.system() == "Linux":
        os.system("mv " + nanoshaper_temp_dir + "*.vert " + output_name + ".vert")
        os.system("mv " + nanoshaper_temp_dir + "*.face " + output_name + ".face")
    elif platform.system() == "Windows":
        os.system("move " + nanoshaper_temp_dir + "*.vert " + output_name + ".vert")
        os.system("move " + nanoshaper_temp_dir + "*.face " + output_name + ".face")
    """
    # Warning de que NanoShaper no creo los archivos
    
  
    
    

    #os.chdir("..")


def convert_msms2off(mesh_face_path, mesh_vert_path, mesh_off_path):
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
    face = open(mesh_face_path, "r").read()
    vert = open(mesh_vert_path, "r").read()

    faces = np.vstack(np.char.split(face.split("\n")[0:-1]))[:, :3].astype(int) - 1
    verts = np.vstack(np.char.split(vert.split("\n")[0:-1]))[:, :3].astype(float)

    # grid = bempp.api.grid_from_element_data(verts.transpose(), faces.transpose())
    grid = bempp.api.Grid(verts.transpose(), faces.transpose())
    return grid


def import_off_mesh(mesh_off_path):
    grid = bempp.api.import_grid(mesh_off_path)
    return grid


def density_to_nanoshaper_grid_scale_conversion(mesh_density):
    grid_scale = round(
        0.797 * (mesh_density ** 0.507), 2
    )  ## Emperical relation found by creatign meshes using nanoshaper and calculating their density
    return grid_scale
