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
    Given a pqr file, import the charges and the coordinates of them. Returns three arrays,
    q with the charges, x_q with the coordinates and r_q with the radii.

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
    >>> q, x_q, r_q = ct.import_charges_from_pqr(pqr_path)
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

    q, x_q, r_q = (
        np.empty((atom_count,)),
        np.empty((atom_count, 3)),
        np.empty((atom_count,)),
    )
    count = 0
    for line in molecule_data:
        line = line.split()
        if len(line) == 0 or line[0] != "ATOM":
            continue
        q[count] = float(line[8])
        x_q[count, :] = line[5:8]
        r_q[count] = float(line[9])

        count += 1

    return q, x_q, r_q


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
        r_q : numpy.array
            Array with the radii of the charges.
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
    q, x_q, r_q = import_charges_from_pqr(mesh_pqr_path)

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
    return grid, q, x_q, r_q


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
        r_q : numpy.array
            Array with the radii of the charges.
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

    q, x_q, r_q = import_charges_from_pqr(mesh_pqr_path)

    return q, x_q, r_q


def read_tinker_radius(filename, radius_keyword='solute', solute_radius_type = 'PB'):
    """
    Read atomic radius from a Tinker prm file.
    Input:
    ------
    filename: (string) prm file name
    radius_keyword: (string) 'vdw' (default) or 'solute' depending on the radius to be used. Radii under the 'solute'
                     keyword have been optimized for implicit solvent calculations
    radius_type: (string) 'PB' (default), 'DDCOSMO' or 'GK'. The radius read from the 'solute' keyword has three alternatives
                     depending on the tool used to optimize.
    """
    file_xyz = filename + ".xyz"
    file_key = filename + ".key"
    
    with open(file_xyz, 'r') as f:
        N = int(f.readline().split()[0])
        
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    r = np.zeros(N)
    
    atom_type  = np.chararray(N, itemsize=10)
    
    i = 0
    header = 0
    
    xyz_file = open(file_xyz, 'r')
    
    for line in xyz_file:
        line = line.split()
        
        if header==1:
            
            x[i] = np.float64(line[2])
            y[i] = np.float64(line[3])
            z[i] = np.float64(line[4])
            
            atom_type[i] = line[5]
            
            i+=1
            
        header = 1
        
    xyz_file.close()
    
    atom_class = {}
    vdw_radii = {}
    solute_radii_PB = {}
    solute_radii_DD = {}
    solute_radii_GK = {}
    
    with open(file_key, 'r') as f:
        
        line = f.readline().split()
        
        if line[0]=='parameters':
            
            file_key = line[1]
            
        print ('Reading parameters from '+file_key)
     
    try: 
        key_file = open(file_key, 'r')
    except OSError:
        print('Cannot find file ' + file_key)
        return
        
    for line in key_file:
        
        line = line.split()
        
        if len(line)>0:
            
            if line[0].lower()=='atom':
                
                atom_class[line[1]] = line[2]
                
            if line[0].lower()=='vdw' and line[1] not in vdw_radii:
                
                vdw_radii[line[1]] = np.float64(line[2])/2.

            if line[0].lower()=='solute' and line[1] not in solute_radii_PB:
                
                solute_radii_PB[line[1]] = np.float64(line[2])/2.
                solute_radii_DD[line[1]] = np.float64(line[3])/2.
                solute_radii_GK[line[1]] = np.float64(line[4])/2.

                
    key_file.close()
                
    for i in range(N):
        
        if radius_keyword == 'vdw':
            r[i] = vdw_radii[atom_class[atom_type[i].decode()]]
        elif radius_keyword == 'solute':
            if solute_radius_type == 'PB':
                r[i] = solute_radii_PB[atom_type[i].decode()]
            elif solute_radius_type == 'DDCOSMO':
                r[i] = solute_radii_DD[atom_type[i].decode()]
            elif solute_radius_type == 'GK':
                r[i] = solute_radii_GK[atom_type[i].decode()]
            else:
                print('Unrecognized solute type radius. Choose between PB, DDCOSMO, and GK.')
                return
        else:
            print('Unrecognized keyword for radius. Choose between vdw and solute.')
            return
        
    return r 

def find_multipole(multipole_list, connections, atom_type, pos, i, N):
#   filter possible multipoles by atom type
    atom_possible = []
    for j in range(len(multipole_list)):
        if atom_type[i] == multipole_list[j][0]:
            atom_possible.append(multipole_list[j])

#   filter possible multipoles by z axis defining atom (needs to be bonded)
#   only is atom_possible has more than 1 alternative
    if len(atom_possible)>1:
        zaxis_possible = []
        for j in range(len(atom_possible)):
            for k in connections[i]:
                neigh_type = atom_type[k]
                if neigh_type == atom_possible[j][1]:
                    zaxis_possible.append(atom_possible[j])

#       filter possible multipoles by x axis defining atom (no need to be bonded)
#       only if zaxis_possible has more than 1 alternative
        if len(zaxis_possible)>1:
            neigh_type = []
            for j in range(len(zaxis_possible)):
                neigh_type.append(zaxis_possible[j][2])

            xaxis_possible_atom = []
            for j in range(N):
                if atom_type[j] in neigh_type and i!=j:
                    xaxis_possible_atom.append(j)

            dist = np.linalg.norm(pos[i,:] - pos[xaxis_possible_atom,:], axis=1)


            xaxis_at_index = np.where(np.abs(dist - np.min(dist))<1e-12)[0][0]
            xaxis_at = xaxis_possible_atom[xaxis_at_index]

#           just check if it's not a connection
            if xaxis_at not in connections[i]:
#                print 'For atom %i+1, x axis define atom is %i+1, which is not bonded'%(i,xaxis_at)
                for jj in connections[i]:
                    if jj in xaxis_possible_atom:
                        print('For atom %i+1, there was a bonded connnection available for x axis, but was not used'%(i))

            xaxis_type = atom_type[xaxis_at]

            xaxis_possible = []
            for j in range(len(zaxis_possible)):
                if xaxis_type == zaxis_possible[j][2]:
                    xaxis_possible.append(zaxis_possible[j])

            if len(xaxis_possible)==0:
                print('For atom %i+1 there is no possible multipole'%i)
            if len(xaxis_possible)>1:
                print ('For atom %i+1 there is more than 1 possible multipole, use last one'%i)

        else:
            xaxis_possible = zaxis_possible

    else:
        xaxis_possible = atom_possible
        
    multipole = xaxis_possible[-1]

    return multipole

def load_tinker_multipoles_to_solute(solute):
    """
    Reads input file from tinker
    Input:
    -----
    solute: (class) 
    Returns:
    -------
    pos: Nx3 array with position of multipoles
    q  : array size N with charges (monopoles)
    p  : array size Nx3 with dipoles
    Q  : array size Nx3x3 with quadrupoles
    alpha: array size Nx3x3 with polarizabilities
            (tinker considers an isotropic value, not tensor)
    """
    filename = solute.xyz_path[:-4] # remove extension
    
    file_xyz = filename+'.xyz'
    file_key = filename+'.key'

    with open(file_xyz, 'r') as f:
        N = int(f.readline().split()[0])
        
    pos   = np.zeros((N,3))
    q     = np.zeros(N)
    p     = np.zeros((N,3))
    Q     = np.zeros((N,3,3))
    alpha = np.zeros((N,3,3))
    thole = np.zeros(N)
    mass  = np.zeros(N)
    atom_type  = np.chararray(N, itemsize=10)
    connections = np.empty(N, dtype=object)
    polar_group = -np.ones(N, dtype=np.int32)
    N_connections = 0
    header = 0
    
    file = open(file_xyz,"r").read().split("\n")
    for line in file:
        line = line.split()
        if header==1 and len(line)>0:
            atom_number = int(line[0])-1
            pos[atom_number,0] = float(line[2])
            pos[atom_number,1] = float(line[3])
            pos[atom_number,2] = float(line[4])
            atom_type[atom_number] = line[5]
            connections[atom_number] = np.zeros(len(line)-6, dtype=int)
            N_connections += len(line)-6
            for i in range(6, len(line)):
                connections[atom_number][i-6] = int(line[i]) - 1
            
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
    
    with open(file_key, 'r') as f:
        line = f.readline().split()
        if line[0]=='parameters':
            file_key = line[1]
            
        if not os.path.isfile(file_key):
            print("Cannot find a key/prm file. Please use same file name as .xyz file.")
            return
    
        print('Reading parameters from '+file_key)
    
    file_k = open(file_key, 'r').read().split('\n') 
    for line in file_k:
        line = line.split()
        
        if len(line)>0:
            if line[0].lower()=='atom':
                atom_class[line[1]] = line[2]
                atom_mass[line[1]] = float(line[-2])
                
            if line[0].lower()=='polarize':
                polarizability[line[1]] = float(line[2])
                thole_factor[line[1]] = float(line[3])
                polar_group_list[line[1]] = np.chararray(len(line)-4, itemsize=10)
                polar_group_list[line[1]][:] = line[4:]
                
            if line[0].lower()=='mpole-12-scale':
                m12scale = float(line[1])
            if line[0].lower()=='mpole-13-scale':
                m13scale = float(line[1])
            if line[0].lower()=='mpole-14-scale':
                m14scale = float(line[1])
            if line[0].lower()=='mpole-15-scale':
                m15scale = float(line[1])
            if line[0].lower()=='polar-12-scale':
                p12scale = float(line[1])
            if line[0].lower()=='polar-13-scale':
                p13scale = float(line[1])
            if line[0].lower()=='polar-14-scale':
                p14scale = float(line[1])
            if line[0].lower()=='polar-15-scale':
                p15scale = float(line[1])
                
            if line[0].lower()=='multipole' or (multipole_flag>0 and multipole_flag<5):
                
                if multipole_flag == 0:
                    key = line[1]
                    z_axis = line[2]
                    x_axis = line[3]
                    
                    if len(line)<5:
                        x_axis = '0'
                    
                    if len(line)>5:
                        y_axis = line[4]
                    else:
                        y_axis = '0'
                        
                    axis_type = 'z_then_x'
                    if float(z_axis)==0:
                        axis_type = 'None'
                    if float(z_axis)!=0 and float(x_axis)==0: 
                        axis_type = 'z_only'
                    if float(z_axis)<0 or float(x_axis)<0:
                        axis_type = 'bisector'
                    if float(x_axis)<0 and float(y_axis)<0: # not implemented yet
                        axis_type = 'z_bisect'
                    if float(z_axis)<0 and float(x_axis)<0 and float(y_axis)<0: # not implemented yet
                        axis_type = '3_fold'
                    
                    # Remove negative defining atom types
                    if z_axis[0]=='-': 
                        z_axis = z_axis[1:]
                    if x_axis[0]=='-': 
                        x_axis = x_axis[1:]
                    if y_axis[0]=='-': 
                        y_axis = y_axis[1:]
                        
                    multipole_list.append((key, z_axis, x_axis, y_axis, axis_type))
                    
                    charge[(key, z_axis, x_axis, y_axis, axis_type)] = float(line[-1])
                if multipole_flag == 1:
                    dipole[(key, z_axis, x_axis, y_axis, axis_type)] = np.array([float(line[0]), float(line[1]), float(line[2])])
                if multipole_flag == 2:
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)] = np.zeros((3,3))
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][0,0] = float(line[0])
                if multipole_flag == 3:
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][1,0] = float(line[0])
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][0,1] = float(line[0])
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][1,1] = float(line[1])
                if multipole_flag == 4:
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][2,0] = float(line[0])
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][0,2] = float(line[0])
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][2,1] = float(line[1])
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][1,2] = float(line[1])
                    quadrupole[(key, z_axis, x_axis, y_axis, axis_type)][2,2] = float(line[2])
                    multipole_flag = -1
                
                multipole_flag += 1
                
    polar_group_counter = 0
    for i in range(N):
#       Get polarizability
        alpha[i,:,:] = np.identity(3)*polarizability[atom_type[i].decode()]
    
#       Get Thole factor
        thole[i] = thole_factor[atom_type[i].decode()]
    
#       Get mass
        mass[i] = atom_mass[atom_type[i].decode()]
    
#       Find atom polarization group
        if polar_group[i]==-1:
#           Check with connections if there is a group member already assigned
            for j in connections[i]:
                if atom_type[j] in polar_group_list[atom_type[i].decode()][:] and polar_group[j]!=-1:
                    if polar_group[i]==-1:
                        polar_group[i] = polar_group[j]
                    elif polar_group[i]!=polar_group[j]:
                        print('double polarization group assigment here!')
#           if no other group members are found, create a new group
            if polar_group[i]==-1:
                polar_group[i] = np.int32(polar_group_counter)
                polar_group_counter += 1
                
#           Now, assign group number to connections in the same group
            for j in connections[i]:
                if atom_type[j] in polar_group_list[atom_type[i].decode()][:]:
                    if polar_group[j]==-1:
                        polar_group[j] = polar_group[i]
                    elif polar_group[j]!=polar_group[i]: 
                        print('double polarization group assigment here too!')
                        
        multipole = find_multipole(multipole_list, connections, atom_type.decode(), pos, i, N)
        
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
            if neigh_type == multipole[2] and k!=z_atom:
                if x_atom == -1:
                    x_atom = k
        
#       Next, look within 1-3 bonded atoms
        if x_atom==-1:
            for k in connections[i]:
                for l in connections[k]:
                    neigh_type = atom_type[l].decode()
                    if neigh_type == multipole[2] and l!=i and l!=z_atom:
                        if x_atom == -1:
                            x_atom = l
                            
#       Else, look within nonbonded atoms
        if x_atom==-1:
            neigh_type = multipole[2]
            x_possible_atom = []
            for j in range(N):
                if atom_type[j] == neigh_type and i != j and j != z_atom:
                    x_possible_atom.append(j)

            if len(x_possible_atom)>0:
                dist = np.linalg.norm(pos[i,:] - pos[x_possible_atom,:], axis=1)

                x_atom_index = np.where(np.abs(dist - np.min(dist))<1e-12)[0][0]
                x_atom = x_possible_atom[x_atom_index]
                
        if x_atom==-1 and multipole[4]=='z_only': # no need for an x_atom
            x_atom = -2    

        if z_atom==-1 or x_atom==-1: # for example, in the sphere case
            i_local = np.array([1,0,0])
            j_local = np.array([0,1,0])
            k_local = np.array([0,0,1])

        else:
            r12 = pos[z_atom,:] - pos[i,:]
            r13 = pos[x_atom,:] - pos[i,:]
            if multipole[4]=='z_then_x':
                k_local = r12/np.linalg.norm(r12) 
                i_local = (r13 - np.dot(r13,k_local)*k_local)/np.linalg.norm(r13 - np.dot(r13,k_local)*k_local)
                j_local = np.cross(k_local, i_local)

            elif multipole[4]=='bisector':
                k_local = r12/np.linalg.norm(r12) + r13/np.linalg.norm(r13) 
                k_local = k_local/np.linalg.norm(k_local)
                i_local = (r13 - np.dot(r13,k_local)*k_local)/np.linalg.norm(r13 - np.dot(r13,k_local)*k_local)
                j_local = np.cross(k_local, i_local)

            elif multipole[4]=='z_only':
                k_local = r12/np.linalg.norm(r12) 
           
                dX = np.array([1.,0.,0.])
                dot = k_local[0]
                if abs(dot) > 0.866:
                    dX[0] = 0.
                    dX[1] = 1.
                    dot = k_local[1]

                dX -= dot*k_local
                i_local = dX/np.linalg.norm(dX)

                j_local = np.cross(k_local, i_local)
                
#       Assign charge
        q[i] = charge[multipole]
    
#       Find rotation matrix
        A = np.identity(3)
        A[:,0] = i_local
        A[:,1] = j_local
        A[:,2] = k_local
        
        bohr = 0.52917721067
#       Assign dipole
        p[i,:] = np.dot(A, dipole[multipole])*bohr
        
#       Assign quadrupole
        for ii in range(3):
            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        Q[i,ii,j] += A[ii,k]*A[j,m]*quadrupole[multipole][k,m]*bohr**2*2 # x2 to agree with Tinker's formulation (they include 1/2 in Q)

#   Connections list
#   1-2 connections (already computed, just put into 1D array)
    connections_12 = np.zeros(N_connections, dtype=np.int32)
    pointer_connections_12 = np.zeros(N+1, dtype=np.int32)    # pointer to beginning of interaction list
    for i in range(N):
        pointer_connections_12[i+1] = pointer_connections_12[i] + len(connections[i])
        start = pointer_connections_12[i]
        stop = pointer_connections_12[i+1]
        connections_12[start:stop] = connections[i]

    if N<2: #if no 1-2 connections
        connections_12 = np.zeros(N) # this avoids a GPU error later
        
#   1-3 connections
    connections_13 = np.zeros(int(N_connections*N_connections/N), dtype=np.int32)
    pointer_connections_13 = np.zeros(N+1, dtype=np.int32)    # pointer to beginning of interaction list
    
    if N>2: # ions and diatomic molecules have no 1-3 connections
        for i in range(N):
            possible_connections = np.concatenate(connections[connections[i]])
            possible_connections = np.unique(possible_connections) # filter out repeated connections 
            index_self = np.where(possible_connections==i)[0] # remove self atom
            possible_connections = np.delete(possible_connections, index_self)
            pointer_connections_13[i+1] = pointer_connections_13[i] + len(possible_connections) 
        
            start = pointer_connections_13[i]
            end   = pointer_connections_13[i+1]
            connections_13[start:end] = possible_connections 

        connections_13 = connections_13[:pointer_connections_13[-1]]
    else:
        connections_13 = np.zeros(N) # this avoids a GPU error later
        
    r = read_tinker_radius(filename, 
                        radius_keyword = solute.radius_keyword, 
                        solute_radius_type = solute.solute_radius_type)

    return pos, q, p, Q, alpha, r, mass, polar_group, thole, \
           connections_12, connections_13, \
           pointer_connections_12, pointer_connections_13, \
           p12scale, p13scale


def generate_msms_mesh_import_tinker_multipoles(solute):
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
        tinker params
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
    data_xyzr = np.zeros((N,4))
    data_xyzr[:,:3] = x_q[:,:]
    data_xyzr[:,3] = r_q[:]
    np.savetxt(mesh_xyzr_path, data_xyzr, fmt='%1.4f')

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
    return grid, x_q, q,d, Q, alpha, r_q, mass, polar_group, thole, \
            connections_12, connections_13, pointer_connections_12, pointer_connections_13, p12scale, p13scale
