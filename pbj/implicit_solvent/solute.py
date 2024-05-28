import re
import bempp.api
import os
import numpy as np
import time
import shutil
import pbj.mesh.mesh_tools as mesh_tools
import pbj.mesh.charge_tools as charge_tools
import pbj.implicit_solvent.pb_formulation.formulations as pb_formulations
import pbj.implicit_solvent.utils as utils


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
        mesh_density=2.0,
        nanoshaper_grid_scale=None,
        solvent_radius=1.4,
        mesh_generator="nanoshaper",
        print_times=False,
        force_field="amber",
        formulation="direct",
        radius_keyword="solute",
        solute_radius_type="PB"
    ):

        if not os.path.isfile(solute_file_path):
            print("file does not exist -> Cannot start")
            return

            
        if force_field == "amoeba" and formulation != "direct":
            print("AMOEBA force field is only available with the direct formulation -> Changing to direct")
        if force_field == "amoeba":
            formulation = "direct_amoeba"

        self._pb_formulation = formulation

        self.formulation_object = getattr(pb_formulations, self.pb_formulation, None)
        if self.formulation_object is None:
            raise ValueError("Unrecognised formulation type %s" % self.pb_formulation)

        self.force_field = force_field

        self.radius_keyword = radius_keyword
        self.solute_radius_type = solute_radius_type

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
        self.mesh_probe_radius = solvent_radius
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

        elif file_extension == "xyz":
            self.imported_file_type = "xyz"
            self.xyz_path = solute_file_path
            self.solute_name = os.path.split(solute_file_path.split(".")[-2])[-1]

        else:
            print("File is not pdb, pqr, or Tinker xyz -> Cannot start")

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

            if force_field == "amoeba":
                (
                    self.x_q, 
                    self.q, 
                    self.d,
                    self.Q, 
                    self.alpha, 
                    self.r_q,
                    self.mass, 
                    self.polar_group, 
                    self.thole, 
                    self.connections_12, 
                    self.connections_13, 
                    self.pointer_connections_12, 
                    self.pointer_connections_13, 
                    self.p12scale, 
                    self.p13scale, 
                ) = charge_tools.load_tinker_multipoles_to_solute(self)

                self.d_induced = np.zeros_like(self.d)
                self.d_induced_prev = np.zeros_like(self.d)
            else: 
                self.q, self.x_q, self.r_q, self.atom_name, self.res_name, self.res_num = charge_tools.load_charges_to_solute(
                    self
                )  # Import charges from given file

        else:  # Generate mesh from given pdb or pqr, and import charges at the same time

            if force_field == "amoeba":
                (
                    self.mesh,
                    self.x_q, 
                    self.q, 
                    self.d, 
                    self.Q, 
                    self.alpha, 
                    self.r_q,
                    self.mass, 
                    self.polar_group, 
                    self.thole, 
                    self.connections_12, 
                    self.connections_13, 
                    self.pointer_connections_12, 
                    self.pointer_connections_13, 
                    self.p12scale, 
                    self.p13scale, 
                ) = charge_tools.generate_msms_mesh_import_tinker_multipoles(self)

                self.d_induced = np.zeros_like(self.d)
                self.d_induced_prev = np.zeros_like(self.d)


            else:
                (
                    self.mesh,
                    self.q,
                    self.x_q,
                    self.r_q,
                    self.atom_name, 
                    self.res_name, 
                    self.res_num
                ) = charge_tools.generate_msms_mesh_import_charges(self)

        self.ep_in = 4.0
        self.ep_ex = 80.0
        self.ep_stern = 80.0
        self.kappa = 0.125

        self.gamma_cav_nonpolar = 0.06
        self.intercept_cav_nonpolar = -3

        self.gamma_disp_nonpolar = -0.055
        self.intercept_disp_nonpolar = 3.5

        self.solvent_number_density = 1.45
        
        self.slic_alpha = 0.5
        self.slic_beta = -60
        self.slic_gamma = -0.5
        
        self.slic_sigma = None
        self.slic_e_hat_diel  = None #self.ep_in / self.ep_stern
        self.slic_e_hat_stern = None #self.ep_stern / self.ep_ex
        
        self.stern_mesh_density_ratio = 0.5 # stern_density/diel_density ratio. No need for fine meshes in Stern.
        self.stern_probe_radius = 0.05 # probe radius for the outer mesh of Stern layer 

        self.sas_mesh_density = self.mesh_density
        
        self.pb_formulation_alpha = 1.0  # np.nan
        self.pb_formulation_beta = self.ep_ex / self.ep_in  # np.nan

        self.pb_formulation_stern_width = 2.0
        self.stern_object = None

        self.pb_formulation_preconditioning = True
        self.pb_formulation_preconditioning_type = "mass_matrix"

        self.discrete_form_type = "weak"

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
        if "preconditioning_matrix_gmres" not in self.matrices: # might already exist if just regenerating RHS
            self.matrices["preconditioning_matrix_gmres"] = None
        if self.formulation_object is None:
            raise ValueError("Unrecognised formulation type %s" % self.pb_formulation)

    @property
    def stern_mesh_density(self):
        return self._stern_mesh_density

    @stern_mesh_density.setter
    def stern_mesh_density(self, value):
        self._stern_mesh_density = value
        self.stern_mesh_density_ratio = value/self.mesh_density
        pb_formulations.direct_stern.create_stern_mesh(self)
        
        
    def display_available_formulations(self):
        from inspect import getmembers, ismodule

        print("Current formulation: " + self.pb_formulation)
        print("List of available formulations:")
        available = getmembers(pb_formulations, ismodule)
        for element in available:
            if element[0] == "common":
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

    def assemble_matrices(self): # not being used, as this is done in apply_preconditioning
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
            self.rhs["rhs_final"] = [rhs for key, rhs in sorted(self.rhs.items())][
                : len(self.matrices["A"].domain_spaces)
            ]

            self.matrices["A_discrete"] = utils.matrix_to_discrete_form(
                self.matrices["A_final"], "weak"
            )
            self.rhs["rhs_discrete"] = utils.rhs_to_discrete_form(
                self.rhs["rhs_final"], "weak", self.matrices["A"]
            )

        self.timings["time_preconditioning"] = time.time() - preconditioning_start_time
        
    def apply_preconditioning_rhs(self):
        preconditioning_start_time = time.time()
        if self.pb_formulation_preconditioning and self.matrices["preconditioning_matrix_gmres"] is None:
            precon_str = self.pb_formulation_preconditioning_type + "_preconditioner_rhs"
            preconditioning_object = getattr(self.formulation_object, precon_str, None)
            if preconditioning_object is not None:
                preconditioning_object(self)
            else:
                raise ValueError(
                    "Unrecognised preconditioning type %s for current formulation type %s"
                    % (self.pb_formulation_preconditioning_type, self.pb_formulation)
                )
        else:
            self.rhs["rhs_final"] = [rhs for key, rhs in sorted(self.rhs.items())][
                : len(self.matrices["A"].domain_spaces)
            ]
            self.rhs["rhs_discrete"] = utils.rhs_to_discrete_form(
                self.rhs["rhs_final"], "weak", self.matrices["A"]
            )

        self.timings["time_preconditioning"] = time.time() - preconditioning_start_time
        
    def calculate_solvation_energy(self, electrostatic_energy=True, nonpolar_energy=False):

        calculate_all = electrostatic_energy and nonpolar_energy
        if calculate_all:
            self.calculate_electrostatic_solvation_energy()
            self.calculate_nonpolar_solvation_energy()
            self.results["solvation_energy"] = self.results["electrostatic_solvation_energy"] \
                                             + self.results["nonpolar_solvation_energy"]

        elif electrostatic_energy:
            self.calculate_electrostatic_solvation_energy()
    
        elif nonpolar_energy: 
            self.calculate_nonpolar_solvation_energy()



    def calculate_electrostatic_solvation_energy(self):

        if "phi" not in self.results:
            print("Please compute surface potential first with simulation.calculate_potentials()")
            return
        
        if self.force_field == "amoeba":
            self.formulation_object.calculate_solvation_energy_polarizable(self)
            return

        start_time = time.time()

        solution_dirichl = self.results["phi"]
        solution_neumann = self.results["d_phi"]

        from bempp.api.operators.potential.laplace import single_layer, double_layer

        slp_q = single_layer(self.neumann_space, self.x_q.transpose())
        dlp_q = double_layer(self.dirichl_space, self.x_q.transpose())
        phi_q = slp_q * solution_neumann - dlp_q * solution_dirichl

        self.results["phir_charges"] = phi_q

        # total solvation energy applying constant to get units [kcal/mol]
        total_energy = 2 * np.pi * 332.064 * np.sum(self.q * phi_q).real
        self.results["electrostatic_solvation_energy"] = total_energy
        self.timings["time_calc_elec_energy"] = time.time() - start_time

        if self.print_times:
            print(
                "It took ",
                self.timings["time_calc_elec_energy"],
                " seconds to compute the electrostatic solvation energy",
            )
 
    def calculate_nonpolar_solvation_energy(self, sas_mesh_density=None):

        start_time = time.time()
        
        self.calculate_cavity_energy(sas_mesh_density)
        self.calculate_dispersion_energy(sas_mesh_density)

        self.timings["time_calc_nonpol_energy"] = time.time() - start_time

        self.results["nonpolar_solvation_energy"] = \
                self.results["cavity_energy"] + self.results["dispersion_energy"]
    
        if self.print_times:
            print(
                "It took ",
                self.timings["time_calc_nonpol_energy"],
                " seconds to compute the nonpolar solvation energy",
            )
 
    def calculate_cavity_energy(self, sas_mesh_density=None):

        if not hasattr(self, "sas_mesh"): 
            self.create_sas_mesh(sas_mesh_density)

        sasa = np.sum(self.sas_mesh.volumes)

        gamma = self.gamma_cav_nonpolar 
        b = self.intercept_cav_nonpolar

        cavity_energy = gamma*sasa + b
        self.results["cavity_energy"] = cavity_energy


    def calculate_dispersion_energy(self, sas_mesh_density=None):

        if not hasattr(self, "sas_mesh"): 
            self.create_sas_mesh(sas_mesh_density)

        sasa = np.sum(self.sas_mesh.volumes)

        gamma = self.gamma_disp_nonpolar 
        b = self.intercept_disp_nonpolar

        dispersion_energy = gamma*sasa + b
        self.results["dispersion_energy"] = dispersion_energy

    def calculate_gradient_field(self, h=0.001):

        """
        Compute the first derivate of potential due to solvent
        in the position of the points
        """

        if "phi" not in self.results:
            print("Please compute surface potential first with simulation.calculate_potentials()")
            return


        start_time = time.time()

        solution_dirichl = self.results["phi"]
        solution_neumann = self.results["d_phi"]

        dphidr = np.zeros([len(self.x_q), 3])
        dist = np.diag([h, h, h])  # matriz 3x3 diagonal de h

        # x axis derivate
        dx = np.concatenate(
            (self.x_q[:] + dist[0], self.x_q[:] - dist[0])
        )  # vector x+h y luego x-h
        slpo = bempp.api.operators.potential.laplace.single_layer(
            self.neumann_space, dx.transpose()
        )
        dlpo = bempp.api.operators.potential.laplace.double_layer(
            self.dirichl_space, dx.transpose()
        )
        phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
        dphidx = 0.5 * (phi[0, : len(self.x_q)] - phi[0, len(self.x_q) :]) / h
        dphidr[:, 0] = dphidx

        # y axis derivate
        dy = np.concatenate((self.x_q[:] + dist[1], self.x_q[:] - dist[1]))
        slpo = bempp.api.operators.potential.laplace.single_layer(
            self.neumann_space, dy.transpose()
        )
        dlpo = bempp.api.operators.potential.laplace.double_layer(
            self.dirichl_space, dy.transpose()
        )
        phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
        dphidy = 0.5 * (phi[0, : len(self.x_q)] - phi[0, len(self.x_q) :]) / h
        dphidr[:, 1] = dphidy

        # z axis derivate
        dz = np.concatenate((self.x_q[:] + dist[2], self.x_q[:] - dist[2]))
        slpo = bempp.api.operators.potential.laplace.single_layer(
            self.neumann_space, dz.transpose()
        )
        dlpo = bempp.api.operators.potential.laplace.double_layer(
            self.dirichl_space, dz.transpose()
        )
        phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
        dphidz = 0.5 * (phi[0, : len(self.x_q)] - phi[0, len(self.x_q) :]) / h
        dphidr[:, 2] = dphidz

        self.results["gradphir_charges"] = dphidr
        self.timings["time_calc_gradient_field"] = time.time() - start_time

        if self.print_times:
            print(
                "It took ",
                self.timings["time_calc_gradient_field"],
                " seconds to compute the gradient field on solute charges",
            )
        return None

    def calculate_gradgradient_field(self, h=0.001):

        """
        Compute the second derivate of potential due to solvent
        in the position of the points
        xq: Array size (Nx3) whit positions to calculate the derivate.
        h: Float number, distance for the central difference.
        Return:
        ddphi: Second derivate of the potential in the positions of points.
        """

        if "phi" not in self.results:
            print("Please compute surface potential first with simulation.calculate_potentials()")
            return

        start_time = time.time()

        x_q = self.x_q
        neumann_space = self.neumann_space
        dirichl_space = self.dirichl_space
        solution_neumann = self.results["d_phi"]
        solution_dirichl = self.results["phi"]

        ddphi = np.zeros((len(x_q),3,3))
        dist = np.array(([h,0,0],[0,h,0],[0,0,h]))
        for i in range(3):
            for j in np.where(np.array([0, 1, 2]) >= i)[0]:
                if i==j:
                    dp = np.concatenate((x_q[:] + dist[i], x_q[:], x_q[:] - dist[i]))
                    slpo = bempp.api.operators.potential.laplace.single_layer(neumann_space, dp.transpose())
                    dlpo = bempp.api.operators.potential.laplace.double_layer(dirichl_space, dp.transpose())
                    phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
                    ddphi[:,i,j] = (phi[0,:len(x_q)] - 2*phi[0,len(x_q):2*len(x_q)] + phi[0, 2*len(x_q):])/(h**2)

                else:
                    dp = np.concatenate((x_q[:] + dist[i] + dist[j], x_q[:] - dist[i] - dist[j], x_q[:] + \
                                         dist[i] - dist[j], x_q[:] - dist[i] + dist[j]))
                    slpo = bempp.api.operators.potential.laplace.single_layer(neumann_space, dp.transpose())
                    dlpo = bempp.api.operators.potential.laplace.double_layer(dirichl_space, dp.transpose())
                    phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
                    ddphi[:,i,j] = (phi[0,:len(x_q)] + phi[0,len(x_q):2*len(x_q)] - \
                                    phi[0, 2*len(x_q):3*len(x_q)] - phi[0, 3*len(x_q):])/(4*h**2)
                    ddphi[:,j,i] = (phi[0,:len(x_q)] + phi[0,len(x_q):2*len(x_q)] - \
                                    phi[0, 2*len(x_q):3*len(x_q)] - phi[0, 3*len(x_q):])/(4*h**2)

            self.results["gradgradphir_charges"] = ddphi
            self.timings["time_calc_gradgrad_field"] = time.time() - start_time

            if self.print_times:
                print(
                    "It took ",
                    self.timings["time_calc_gradgrad_field"],
                    " seconds to compute the gradient of the gradient field on solute charges",
                )


    def calculate_charges_forces(self, h=0.001):

        if "phi" not in self.results:
            print("Please compute surface potential first with simulation.calculate_potentials()")
            return

        if "gradphir_charges" not in self.results:
            # If gradient field has not been calculated, calculate it now
            self.calculate_gradient_field(h=h)

        start_time = time.time()

        dphidr = self.results["gradphir_charges"]

        convert_to_kcalmolA = 4 * np.pi * 332.0636817823836

        f_reac = convert_to_kcalmolA * -np.transpose(np.transpose(dphidr) * self.q)
        f_reactotal = np.sum(f_reac, axis=0)

        self.results["f_qf_charges"] = f_reac
        self.results["f_qf"] = f_reactotal
        self.timings["time_calc_solute_force"] = time.time() - start_time

        if self.print_times:
            print(
                "It took ",
                self.timings["time_calc_solute_force"],
                " seconds to compute the force on solute charges",
            )

        return None

    def calculate_boundary_forces(self, fdb_approx=False):

        if "phi" not in self.results:
            print("Please compute surface potential first with simulation.calculate_potentials()")
            return

        start_time = time.time()

        phi = self.results["phi"].evaluate_on_element_centers()
        d_phi = self.results["d_phi"].evaluate_on_element_centers()

        convert_to_kcalmolA = 4 * np.pi * 332.0636817823836
        dS = np.transpose(np.transpose(self.mesh.normals) * self.mesh.volumes)

        if fdb_approx :
            # Dielectric boundary force
            f_db = (
                -0.5
                * convert_to_kcalmolA
                * (self.ep_ex - self.ep_in)
                * (self.ep_in / self.ep_ex)
                * np.sum(np.transpose(np.transpose(dS) * d_phi[0] ** 2), axis=0)
            )

        else:
            N_elements = self.mesh.number_of_elements
            phi_vertex = self.results["phi"].coefficients
            ep_hat =  self.ep_in/self.ep_ex
            dphi_centers = ep_hat * self.results["d_phi"].evaluate_on_element_centers()[0]
            f_db = np.zeros(3)
            convert_to_kcalmolA = 4 * np.pi * 332.0636817823836
            for i in range(N_elements):
                eps = self.mesh.normals[i]
                    
                # get vertex indices adyacent to a triangular element
                v1_index = self.mesh.elements[0,i]
                v2_index = self.mesh.elements[1,i]
                v3_index = self.mesh.elements[2,i]
                
                # get vertex coordinates from vertex indices
                v1 = self.mesh.vertices[:,v1_index]
                v2 = self.mesh.vertices[:,v2_index]
                v3 = self.mesh.vertices[:,v3_index]
                
                v21 = v2 - v1
                v31 = v3 - v1
                
                v21_norm = np.linalg.norm(v21)
                v31_norm = np.linalg.norm(v31)
                
                phi_1 = phi_vertex[v1_index]
                phi_2 = phi_vertex[v2_index]
                phi_3 = phi_vertex[v3_index]
                
                alpha = np.arccos(np.dot(v21,v31)/(v21_norm*v31_norm))
                
                a = (phi_2 - phi_1)/v21_norm
                b = (phi_3 - phi_1)/(v31_norm*np.sin(alpha)) - (phi_2-phi_1)/(v21_norm*np.tan(alpha))
                
                eta = v21/v21_norm
                tau = np.cross(eps,eta)
                
                E_eps = -dphi_centers[i]
                E_eta = -a
                E_tau = -b
                

                F = ((1/ep_hat)*E_eps*E_eps + E_eta*E_eta + E_tau*E_tau)
                F *= -0.5*(self.ep_ex - self.ep_in)*self.mesh.normals[i] * self.mesh.volumes[i]
                
                f_db += convert_to_kcalmolA * F    

        # Ionic boundary force
        f_ib = (
            -0.5
            * convert_to_kcalmolA
            * (self.ep_ex)
            * (self.kappa**2)
            * np.sum(np.transpose(np.transpose(dS) * phi[0] ** 2), axis=0)
        )

        self.results["f_db"] = f_db
        self.results["f_ib"] = f_ib
        self.timings["time_calc_boundary_force"] = time.time() - start_time

        if self.print_times:
            print(
                "It took ",
                self.timings["time_calc_boundary_force"],
                " seconds to compute the boundary forces",
            )

    def calculate_solvation_forces(self, h=0.001, force_formulation='maxwell_tensor', fdb_approx=False):

        if "phi" not in self.results:
            print("Please compute surface potential first with simulation.calculate_potentials()")
            return

        if force_formulation == 'energy_functional':
            if "f_qf" not in self.results:
                self.calculate_gradient_field(h=h)
                self.calculate_charges_forces()

            self.calculate_boundary_forces(fdb_approx=fdb_approx)

            start_time = time.time()

            f_solv = np.zeros([3])
            f_qf = self.results["f_qf"]
            f_db = self.results["f_db"]
            f_ib = self.results["f_ib"]
            f_solv = f_qf + f_db + f_ib

            self.results["f_solv"] = f_solv
            self.timings["time_calc_solvation_force"] = (
                time.time()
                - start_time
                + self.timings["time_calc_boundary_force"]
                + self.timings["time_calc_solute_force"]
                + self.timings["time_calc_gradient_field"]
            )
            if self.print_times:
                print(
                    "It took ",
                    self.timings["time_calc_solvation_force"],
                    " seconds to compute the solvation forces with ",
                    force_formulation,
                    " formulation"
                )

        elif force_formulation == 'maxwell_tensor':

            if "f_ib" not in self.results:
                self.calculate_boundary_forces()

            start_time = time.time()


            N_elements = self.mesh.number_of_elements
            P_normal = np.zeros([N_elements])
            phi_vertex = self.results["phi"].coefficients
            ep_hat =  self.ep_in/self.ep_ex
            dphi_centers = ep_hat * self.results["d_phi"].evaluate_on_element_centers()[0]
            total_force = np.zeros(3)
            convert_to_kcalmolA = 4 * np.pi * 332.0636817823836

            for i in range(N_elements):
                eps = self.mesh.normals[i]
                
                # get vertex indices adyacent to a triangular element
                v1_index = self.mesh.elements[0,i]
                v2_index = self.mesh.elements[1,i]
                v3_index = self.mesh.elements[2,i]
                
                # get vertex coordinates from vertex indices
                v1 = self.mesh.vertices[:,v1_index]
                v2 = self.mesh.vertices[:,v2_index]
                v3 = self.mesh.vertices[:,v3_index]
                
                v21 = v2 - v1
                v31 = v3 - v1
                
                v21_norm = np.linalg.norm(v21)
                v31_norm = np.linalg.norm(v31)
                
                phi_1 = phi_vertex[v1_index]
                phi_2 = phi_vertex[v2_index]
                phi_3 = phi_vertex[v3_index]
                
                alpha = np.arccos(np.dot(v21,v31)/(v21_norm*v31_norm))
                
                a = (phi_2 - phi_1)/v21_norm
                b = (phi_3 - phi_1)/(v31_norm*np.sin(alpha)) - (phi_2-phi_1)/(v21_norm*np.tan(alpha))
                
                eta = v21/v21_norm
                tau = np.cross(eps,eta)
                
                E_eps = -dphi_centers[i]
                E_eta = -a
                E_tau = -b
                
                E_norm = np.sqrt(E_eps*E_eps + E_eta*E_eta + E_tau*E_tau)
                
                F = (E_eps*E_eps - 0.5*E_norm*E_norm)*eps + E_eps*E_eta*eta + E_eps*E_tau*tau 
                
                F *= self.ep_ex
                total_force += F * self.mesh.volumes[i]   
                P_normal[i] = np.sqrt(np.dot(F * self.mesh.volumes[i], F * self.mesh.volumes[i]))
                
            self.results["P_normal"] = convert_to_kcalmolA * P_normal
            self.results["f_solv"] = convert_to_kcalmolA * total_force + self.results["f_ib"]
            self.timings["time_calc_solvation_force"] = (
                time.time()
                - start_time
            )
            if self.print_times:
                print(
                    "It took ",
                    self.timings["time_calc_solvation_force"],
                    " seconds to compute the solvation forces with ",
                    force_formulation,
                    " formulation"
                )

        else:
            raise ValueError('Formulation have to be "maxwell_tensor" or "energy_functional"')
            
            
    def calculate_coulomb_potential(self, eval_points):
        """
        Compute the Coulomb potential due to the charges in self on eval_points 
        Inputs:
        -------
            eval_points: (Nx3 array) positions where to compute the potential
            
        Output:
        --------
            phi_coul (array) Coulomb potential at eval_points
        """
        
        phi_coul = np.zeros(len(eval_points), dtype=float)
        
        for i in range(len(self.x_q)):
            dist = np.linalg.norm(eval_points - self.x_q[i,:], axis=1)
            phi_coul[:] += self.q[i]/(4*np.pi*dist[:])
            
        return phi_coul
                               
    
    def create_sas_mesh(self, sas_mesh_density=None):

        if sas_mesh_density!=None:
            self.sas_mesh_density = sas_mesh_density

        sas_mesh_dir = os.path.abspath("mesh_files/")
        if self.save_mesh_build_files:
            sas_mesh_dir = self.mesh_build_files_dir

        if not os.path.exists(sas_mesh_dir):
            try:
                os.mkdir(sas_mesh_dir)
            except OSError:
                print("Creation of the directory %s failed" % sas_mesh_dir)

        sas_pqr_file = os.path.join(sas_mesh_dir, "sas_pqr.pqr")
        with open(sas_pqr_file, "w") as f:
            f.write(
                "# This is a dummy pqr file generated for the creation of the SAS mesh.\n"
            )
            for index in range(len(self.r_q)):
                f.write(
                    "ATOM      #  #   ###     #      "
                    + str(self.x_q[index][0])
                    + " "
                    + str(self.x_q[index][1])
                    + " "
                    + str(self.x_q[index][2])
                    + " "
                    + str(self.q[index])
                    + " "
                    + str(self.r_q[index] + self.mesh_probe_radius)
                    + "\n"
                )

        sas_mesh_xyzr_file = os.path.join(sas_pqr_file[:-4] + ".xyzr")
        mesh_tools.convert_pqr2xyzr(sas_pqr_file, sas_mesh_xyzr_file)
    
        probe_radius = 0.05 # small probe for SAS
        if self.mesh_generator == "msms":
            mesh_tools.generate_msms_mesh(
                sas_mesh_xyzr_file,
                sas_mesh_dir,
                self.solute_name+"_sas",
                self.sas_mesh_density,
                probe_radius,
            ) 

        if self.mesh_generator == "nanoshaper":
            nanoshaper_grid_scale =  \
                    mesh_tools.density_to_nanoshaper_grid_scale_conversion(self.sas_mesh_density)		
            mesh_tools.generate_nanoshaper_mesh(
				sas_mesh_xyzr_file,
				sas_mesh_dir,
				self.solute_name+"_sas",
				nanoshaper_grid_scale,
			    probe_radius,
				self.save_mesh_build_files,
			) 

        mesh_face_path = os.path.join(sas_mesh_dir, self.solute_name + "_sas.face")
        mesh_vert_path = os.path.join(sas_mesh_dir, self.solute_name + "_sas.vert")

        grid = mesh_tools.import_msms_mesh(mesh_face_path, mesh_vert_path) 

        if not self.save_mesh_build_files:
            shutil.rmtree(sas_mesh_dir)

        self.sas_mesh = grid
     
            

def get_name_from_pdb(pdb_path):
    pdb_file = open(pdb_path)
    first_line = pdb_file.readline()
    first_line_split = re.split(r"\s{2,}", first_line)
    solute_name = first_line_split[3].lower()
    pdb_file.close()

    return solute_name
