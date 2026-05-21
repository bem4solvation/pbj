import re
import bempp_cl as bempp
import bempp_cl.api
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
        solute_radius_type="PB",
    ):
        """Initializes the Solute object, sets up simulation parameters,
        and handles mesh/charge loading.

        Args:
            solute_file_path (str): Path to the molecular structure file (.pdb, .pqr, or .xyz).
            external_mesh_file (str, optional): Path to a pre-computed mesh file.
                If no extension is given, MSMS (.face/.vert) is assumed. Defaults to None.
            save_mesh_build_files (bool, optional): If True, retains intermediate files
                created during the mesh generation process. Defaults to False.
            mesh_build_files_dir (str, optional): Directory where the intermediate mesh
                files will be stored. Defaults to "mesh_files/".
            mesh_density (float, optional): Density of the vertices for the generated molecular mesh.
                Defaults to 2.0.
            nanoshaper_grid_scale (float, optional): Specific grid scale for NanoShaper.
                If None, it's calculated from mesh_density. Defaults to None.
            solvent_radius (float, optional): Radius of the solvent probe molecule in Angstroms.
                Defaults to 1.4.
            mesh_generator (str, optional): Software to use for mesh generation
                (e.g., "nanoshaper" or "msms"). Defaults to "nanoshaper".
            print_times (bool, optional): If True, prints detailed execution timings
                for benchmarking. Defaults to False.
            force_field (str, optional): Force field model to apply (e.g., "amber", "amoeba").
                Defaults to "amber".
            formulation (str, optional): Electrostatic formulation type to be used
                by pb_formulations. Defaults to "direct".
            radius_keyword (str, optional): Keyword identifier for atom radii selection.
                Defaults to "solute".
            solute_radius_type (str, optional): Target type for the solute atom radii classification.
                Defaults to "PB".

        Raises:
            ValueError: If the specified formulation does not match any available
                module in pb_formulations.
        """

        if not os.path.isfile(solute_file_path):
            print("file does not exist -> Cannot start")
            return

        if force_field == "amoeba" and formulation != "direct":
            print(
                "AMOEBA force field is only available with the direct formulation -> Changing to direct"
            )
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
                (
                    self.q,
                    self.x_q,
                    self.r_q,
                    self.atom_name,
                    self.res_name,
                    self.res_num,
                ) = charge_tools.load_charges_to_solute(
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
                    self.res_num,
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
        self.slic_e_hat_diel = None  # self.ep_in / self.ep_stern
        self.slic_e_hat_stern = None  # self.ep_stern / self.ep_ex

        self.stern_mesh_density_ratio = (
            0.5  # stern_density/diel_density ratio. No need for fine meshes in Stern.
        )
        self.stern_probe_radius = 0.05  # probe radius for the outer mesh of Stern layer

        if nanoshaper_grid_scale is None:
            self.sas_mesh_density = self.mesh_density
        else:
            self.sas_mesh_density = self.nanoshaper_grid_scale

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
        if (
            "preconditioning_matrix_gmres" not in self.matrices
        ):  # might already exist if just regenerating RHS
            self.matrices["preconditioning_matrix_gmres"] = None
        if self.formulation_object is None:
            raise ValueError("Unrecognised formulation type %s" % self.pb_formulation)

    @property
    def stern_mesh_density(self):
        return self._stern_mesh_density

    @stern_mesh_density.setter
    def stern_mesh_density(self, value):
        self._stern_mesh_density = value
        self.stern_mesh_density_ratio = value / self.sas_mesh_density
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

    def assemble_matrices(
        self,
    ):
        # not being used, as this is done in apply_preconditioning
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
        r"""Apply preconditioning to the boundary element matrix operator and system right-hand side.

        Transforms the continuous linear system layout to optimize iterative solver (e.g., GMRES)
        convergence behavior. If preconditioning is active, this method dynamically delegates
        the operator transformations to the active `formulation_object`. Otherwise, it directly
        discretizes the system using standard weak-form boundary element mappings:

        $$ \mathbf{A}_{\text{discrete}} = \text{utils.matrix\_to\_discrete\_form}(\mathbf{A}_{\text{final}}, \text{"weak"}) $$
        $$ \mathbf{b}_{\text{discrete}} = \text{utils.rhs\_to\_discrete\_form}(\mathbf{b}_{\text{final}}, \text{"weak"}, \mathbf{A}) $$

        Raises:
            ValueError: If the requested preconditioning type string does not map to a valid
                        attribute on the active `formulation_object`.

        Side Effects:
            - Dynamically executes `<type>_preconditioner(self)` on `self.formulation_object`.
            - Modifies the `self.matrices` and `self.rhs` dictionaries when no preconditioning is applied.
            - Updates execution profiling timestamps inside `self.timings["time_preconditioning"]`.
        """
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
        r"""Apply preconditioning to the right-hand side (RHS) vector or prepare the discrete standard RHS.

        Transforms the continuous boundary element right-hand side vectors to match the
        selected linear system preconditioning layout. If preconditioning is enabled, this
        method dynamically invokes the formulation-specific RHS preprocessing callback.
        Otherwise, it falls back to a standard weak-form discretization matching the block
        structure of the system matrix:

        $$ \mathbf{b}_{\text{discrete}} = \text{utils.rhs\_to\_discrete\_form}(\mathbf{b}_{\text{final}}, \text{"weak"}, \mathbf{A}) $$

        Raises:
            ValueError: If the requested preconditioning type string does not map to a valid
                        attribute on the active `formulation_object`.

        Side Effects:
            - Dynamically executes `<type>_preconditioner_rhs(self)` on `self.formulation_object`.
            - Modifies `self.rhs["rhs_final"]` and `self.rhs["rhs_discrete"]` when no
              preconditioning is applied.
            - Updates execution profiling timestamps inside `self.timings["time_preconditioning"]`.
        """
        preconditioning_start_time = time.time()
        if (
            self.pb_formulation_preconditioning
            and self.matrices["preconditioning_matrix_gmres"] is None
        ):
            precon_str = (
                self.pb_formulation_preconditioning_type + "_preconditioner_rhs"
            )
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

    def calculate_solvation_energy(
        self, electrostatic_energy=True, nonpolar_energy=False, units="kcal_mol"
    ):
        r"""Calculate the total solvation free energy of the solute.

        Compute and combine the polar (electrostatic) and nonpolar components of the implicit solvation
        free energy based on the active flags:

        $$ \Delta G_{\text{solv}} = \Delta G_{\text{electrostatic}} + \Delta G_{\text{nonpolar}} $$

        Args:
            electrostatic_energy (bool, optional): If True, computes the electrostatic/polar
                                                 solvation energy component. Defaults to True.
            nonpolar_energy (bool, optional): If True, computes the nonpolar (cavity + dispersion)
                                              solvation energy component. Defaults to False.

        Side Effects:
            - Triggers `self.calculate_electrostatic_solvation_energy()` if `electrostatic_energy` is True.
            - Triggers `self.calculate_nonpolar_solvation_energy()` if `nonpolar_energy` is True.
            - Modifies `self.results["solvation_energy"]` to store the cumulative sum when both components are requested.
        """

        calculate_all = electrostatic_energy and nonpolar_energy
        if calculate_all:
            self.calculate_electrostatic_solvation_energy(units=units)
            self.calculate_nonpolar_solvation_energy(units=units)
            self.results["solvation_energy"] = (
                self.results["electrostatic_solvation_energy"]
                + self.results["nonpolar_solvation_energy"]
            )

        elif electrostatic_energy:
            self.calculate_electrostatic_solvation_energy(units=units)

        elif nonpolar_energy:
            self.calculate_nonpolar_solvation_energy(units=units)

    def calculate_electrostatic_solvation_energy(self, units="kcal_mol"):
        r"""Calculate the electrostatic component of the solvation free energy.

        Computes the electrostatic reaction potential ($\phi_{\text{reac}}$) at each explicit
        solute point charge location by projecting the boundary element solution (Dirichlet
        and Neumann data) back into the solute cavity using Laplace potential operators. It then
        evaluates the total polar solvation energy via the charge-potential product:

        $$ E_{\text{electrostatic}} = \frac{1}{2} \sum_{i} q_i \phi_{\text{reac}}(\mathbf{r}_i) $$

        If the polarizable AMOEBA force field is active, the calculation is automatically
        delegated to a specialized polarizable energy formulation handler.

        Side Effects:
            - Modifies `self.results["phir_charges"]` to store the reaction potential evaluated at each charge.
            - Modifies `self.results["electrostatic_solvation_energy"]` to store the total polar energy in kcal/mol.
            - Updates execution profiling timestamps inside `self.timings`.
            - Prints processing time information to standard output if `self.print_times` is True.
        """

        if "phi" not in self.results:
            print(
                "Please compute surface potential first with simulation.calculate_potentials()"
            )
            return

        if self.force_field == "amoeba":
            print("Defaults units for AMOEBA calculations are kcal/mol, ignoring input units argument.")
            self.formulation_object.calculate_solvation_energy_polarizable(self)
            return

        start_time = time.time()

        solution_dirichl = self.results["phi"]
        solution_neumann = self.results["d_phi"]

        from bempp_cl.api.operators.potential.laplace import single_layer, double_layer

        slp_q = single_layer(self.neumann_space, self.x_q.transpose())
        dlp_q = double_layer(self.dirichl_space, self.x_q.transpose())
        phi_q = slp_q * solution_neumann - dlp_q * solution_dirichl

        self.results["phir_charges"] = phi_q

        # total solvation energy applying constant to get units [kcal/mol]
        factor_units = convert_units(units)
        total_energy = 0.5 * factor_units * np.sum(self.q * phi_q).real
        self.results["electrostatic_solvation_energy"] = total_energy
        self.timings["time_calc_elec_energy"] = time.time() - start_time

        if self.print_times:
            print(
                "It took ",
                self.timings["time_calc_elec_energy"],
                " seconds to compute the electrostatic solvation energy",
            )

    def calculate_nonpolar_solvation_energy(
        self, sas_mesh_density=None, units="kcal_mol"
    ):
        r"""Calculate the total nonpolar solvation free energy contribution.

        Combines the energy required to form the molecular cavity in the solvent
        with the attractive van der Waals dispersion interactions between the solute
        and surrounding solvent molecules:

        $$ E_{\text{nonpolar}} = E_{\text{cavity}} + E_{\text{dispersion}} $$

        Args:
            sas_mesh_density (float, optional): Density parameter passed down to the
                                                SAS mesh generator if the mesh has not
                                                yet been constructed. Defaults to None.

        Side Effects:
            - Triggers `self.calculate_cavity_energy(sas_mesh_density, units=units)`.
            - Triggers `self.calculate_dispersion_energy(sas_mesh_density, units=units)`.
            - Modifies `self.results["nonpolar_solvation_energy"]` to store the combined sum.
            - Updates execution profiling timestamps inside `self.timings`.
            - Prints processing time information to standard output if `self.print_times` is True.
        """

        start_time = time.time()

        self.calculate_cavity_energy(sas_mesh_density, units=units)
        self.calculate_dispersion_energy(sas_mesh_density, units=units)

        self.timings["time_calc_nonpol_energy"] = time.time() - start_time

        self.results["nonpolar_solvation_energy"] = (
            self.results["cavity_energy"] + self.results["dispersion_energy"]
        )

        if self.print_times:
            print(
                "It took ",
                self.timings["time_calc_nonpol_energy"],
                " seconds to compute the nonpolar solvation energy",
            )

    def calculate_cavity_energy(self, sas_mesh_density=None, units="kcal_mol"):
        r"""Calculate the nonpolar cavity formation energy based on the Solvent Accessible Surface Area (SASA).

        Computes the reversible work required to create a solute-sized cavity in the
        solvent. This nonpolar component is modeled via a linear relationship with the SASA:

        $$ E_{\text{cav}} = \gamma \cdot \text{SASA} + b $$

        The SASA is determined by summing the surface area elements (`volumes`) of the SAS mesh.

        Args:
            sas_mesh_density (float, optional): Density parameter passed to `create_sas_mesh`
                                                if the SAS mesh hasn't been generated yet.
                                                Defaults to None.

        Side Effects:
            - If `self.sas_mesh` is missing, triggers `self.create_sas_mesh(sas_mesh_density)`.
            - Modifies `self.results["cavity_energy"]` to store the final energy calculation.
        """

        if not hasattr(self, "sas_mesh"):
            self.create_sas_mesh(sas_mesh_density)

        sasa = np.sum(self.sas_mesh.volumes)

        gamma = self.gamma_cav_nonpolar
        b = self.intercept_cav_nonpolar

        cavity_energy = gamma * sasa + b
        factor_units = convert_units(units) / convert_units("kcal_mol")
        self.results["cavity_energy"] = factor_units * cavity_energy

    def calculate_dispersion_energy(self, sas_mesh_density=None, units="kcal_mol"):
        r"""Calculate the nonpolar dispersion energy based on the Solvent Accessible Surface Area (SASA).

        Computes the hydrophobic/nonpolar dispersion contribution to the solvation free
        energy using a linear relationship with the SASA ($E_{\text{disp}} = \gamma \cdot \text{SASA} + b$).
        The SASA is determined by summing the individual element areas (`volumes`) of the SAS mesh.

        Args:
            sas_mesh_density (float, optional): Density parameter passed to `create_sas_mesh`
                                                if the SAS mesh hasn't been generated yet.
                                                Defaults to None.

        Side Effects:
            - If `self.sas_mesh` is missing, triggers `self.create_sas_mesh(sas_mesh_density)`
              to generate it.
            - Modifies `self.results["dispersion_energy"]` to store the final energy calculation.
        """

        if not hasattr(self, "sas_mesh"):
            self.create_sas_mesh(sas_mesh_density)

        sasa = np.sum(self.sas_mesh.volumes)

        gamma = self.gamma_disp_nonpolar
        b = self.intercept_disp_nonpolar

        dispersion_energy = gamma * sasa + b
        factor_units = convert_units(units) / convert_units("kcal_mol")
        self.results["dispersion_energy"] = factor_units * dispersion_energy

    def calculate_gradient_field(self, h=0.001):
        r"""Compute the gradient vector (first-order spatial derivatives) of the reaction potential.

        Evaluates the electric field gradient induced by the solvent at each solute charge
        position ($\nabla \phi_{\text{reac}}$). The method utilizes a second-order central
        finite difference approximation ($\pm h$) along the X, Y, and Z axes using Bempp's
        Laplace single-layer and double-layer boundary potential operators.

        Args:
            h (float, optional): The displacement step size used for the central
                                 finite difference approximation. Defaults to 0.001.

        Returns:
            None

        Side Effects:
            - Modifies `self.results["gradphir_charges"]` to store an $N \times 3$ array
            - Updates execution profiling timestamps inside `self.timings`.
            - Prints processing time information to standard output if `self.print_times` is True.
        """
        if "phi" not in self.results:
            print(
                "Please compute surface potential first with simulation.calculate_potentials()"
            )
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
        r"""Compute the Hessian matrix (second spatial derivatives) of the reaction potential.

        Evaluates the second-order partial derivatives of the electrostatic potential
        induced by the solvent at each solute charge position ($\nabla^2 \phi_{\text{reac}}$).
        The method uses a second-order central finite difference scheme by querying
        Bempp's Laplace single-layer and double-layer potential operators at spatially
        shifted coordinates.

        Args:
            h (float, optional): The displacement step size used for the central
                                 finite difference approximation. Defaults to 0.001.

        Side Effects:
            - Modifies `self.results["gradgradphir_charges"]` to store an $N \times 3 \times 3$
              array representing the full Hessian matrix for each of the $N$ point charges.
            - Updates execution profiling timestamps inside `self.timings`.
            - Prints processing time information to standard output if `self.print_times` is True.
        """

        if "phi" not in self.results:
            print(
                "Please compute surface potential first with simulation.calculate_potentials()"
            )
            return

        start_time = time.time()

        x_q = self.x_q
        neumann_space = self.neumann_space
        dirichl_space = self.dirichl_space
        solution_neumann = self.results["d_phi"]
        solution_dirichl = self.results["phi"]

        ddphi = np.zeros((len(x_q), 3, 3))
        dist = np.array(([h, 0, 0], [0, h, 0], [0, 0, h]))
        for i in range(3):
            for j in np.where(np.array([0, 1, 2]) >= i)[0]:
                if i == j:
                    dp = np.concatenate((x_q[:] + dist[i], x_q[:], x_q[:] - dist[i]))
                    slpo = bempp.api.operators.potential.laplace.single_layer(
                        neumann_space, dp.transpose()
                    )
                    dlpo = bempp.api.operators.potential.laplace.double_layer(
                        dirichl_space, dp.transpose()
                    )
                    phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(
                        solution_dirichl
                    )
                    ddphi[:, i, j] = (
                        phi[0, : len(x_q)]
                        - 2 * phi[0, len(x_q) : 2 * len(x_q)]
                        + phi[0, 2 * len(x_q) :]
                    ) / (h**2)

                else:
                    dp = np.concatenate(
                        (
                            x_q[:] + dist[i] + dist[j],
                            x_q[:] - dist[i] - dist[j],
                            x_q[:] + dist[i] - dist[j],
                            x_q[:] - dist[i] + dist[j],
                        )
                    )
                    slpo = bempp.api.operators.potential.laplace.single_layer(
                        neumann_space, dp.transpose()
                    )
                    dlpo = bempp.api.operators.potential.laplace.double_layer(
                        dirichl_space, dp.transpose()
                    )
                    phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(
                        solution_dirichl
                    )
                    ddphi[:, i, j] = (
                        phi[0, : len(x_q)]
                        + phi[0, len(x_q) : 2 * len(x_q)]
                        - phi[0, 2 * len(x_q) : 3 * len(x_q)]
                        - phi[0, 3 * len(x_q) :]
                    ) / (4 * h**2)
                    ddphi[:, j, i] = (
                        phi[0, : len(x_q)]
                        + phi[0, len(x_q) : 2 * len(x_q)]
                        - phi[0, 2 * len(x_q) : 3 * len(x_q)]
                        - phi[0, 3 * len(x_q) :]
                    ) / (4 * h**2)

            self.results["gradgradphir_charges"] = ddphi
            self.timings["time_calc_gradgrad_field"] = time.time() - start_time

            if self.print_times:
                print(
                    "It took ",
                    self.timings["time_calc_gradgrad_field"],
                    " seconds to compute the gradient of the gradient field on solute charges",
                )

    def calculate_charges_forces(self, h=0.001, units="kcal_molA"):
        r"""Calculate the electrostatic fixed-charge reaction forces acting directly on the solute charges.

        Computes the force exerted on each individual point charge within the solute due to
        the gradient of the reaction potential ($\nabla \phi_{\text{reac}}$). It then sums
        these components to obtain the total fixed-charge force ($f_{qf}$), scaling the final
        output to kcal/mol/Å.

        $$f_{qf} = \sum_{i} -q_i \\nabla \phi_{\text{reac}}(\mathbf{r}_i)$$

        Args:
            h (float, optional): Finite difference step size passed to `calculate_gradient_field`
                                 if the reaction field gradient hasn't been computed yet.
                                 Defaults to 0.001.

        Side Effects:
            - Modifies `self.results["f_qf_charges"]` to store the 3D force vector for each charge.
            - Modifies `self.results["f_qf"]` to store the cumulative 3D force vector.
            - Updates execution profiling timestamps inside `self.timings`.
            - Prints processing time information to standard output if `self.print_times` is True.
        """
        if "phi" not in self.results:
            print(
                "Please compute surface potential first with simulation.calculate_potentials()"
            )
            return

        if "gradphir_charges" not in self.results:
            # If gradient field has not been calculated, calculate it now
            self.calculate_gradient_field(h=h)

        start_time = time.time()

        dphidr = self.results["gradphir_charges"]

        factor_units = convert_units(units)

        f_reac = factor_units * -np.transpose(np.transpose(dphidr) * self.q)
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

    def calculate_boundary_forces(self, fdb_approx=False, units="kcal_molA"):
        """Calculate dielectric and ionic boundary forces (energy functional approach)
        acting on the solute interface.
        (change units conversion to kcal/mol/Å)

        Computes the dielectric boundary force ($f_{db}$) and the ionic boundary force
        ($f_{ib}$) components of the total solvation force. The dielectric force can
        either be evaluated using a exact approach or a normal approximation (fdb_approx).

        Args:
            fdb_approx (bool, optional): If True, computes the dielectric boundary force
                                         using a simplified approximation based solely on
                                         the normal derivative of the potential ($d_phi$).
                                         If False, performs a comprehensive surface element
                                         gradient calculation. Defaults to False.

        Side Effects:
            - Modifies `self.results["f_db"]` with the 3D dielectric boundary force vector.
            - Modifies `self.results["f_ib"]` with the 3D ionic boundary force vector.
            - Updates execution profiling timestamps inside `self.timings`.
            - Prints processing time information to standard output if `self.print_times` is True.
        """

        if "phi" not in self.results:
            print(
                "Please compute surface potential first with simulation.calculate_potentials()"
            )
            return

        start_time = time.time()

        phi = self.results["phi"].evaluate_on_element_centers()
        d_phi = self.results["d_phi"].evaluate_on_element_centers()

        factor_units = convert_units(units)
        dS = np.transpose(np.transpose(self.mesh.normals) * self.mesh.volumes)

        if fdb_approx:
            # Dielectric boundary force
            f_db = (
                -0.5
                * factor_units
                * (self.ep_ex - self.ep_in)
                * (self.ep_in / self.ep_ex)
                * np.sum(np.transpose(np.transpose(dS) * d_phi[0] ** 2), axis=0)
            )

        else:
            N_elements = self.mesh.number_of_elements
            phi_vertex = self.results["phi"].coefficients
            ep_hat = self.ep_in / self.ep_ex
            dphi_centers = (
                ep_hat * self.results["d_phi"].evaluate_on_element_centers()[0]
            )
            f_db = np.zeros(3)
            for i in range(N_elements):
                # eps = self.mesh.normals[i]

                # get vertex indices adyacent to a triangular element
                v1_index = self.mesh.elements[0, i]
                v2_index = self.mesh.elements[1, i]
                v3_index = self.mesh.elements[2, i]

                # get vertex coordinates from vertex indices
                v1 = self.mesh.vertices[:, v1_index]
                v2 = self.mesh.vertices[:, v2_index]
                v3 = self.mesh.vertices[:, v3_index]

                v21 = v2 - v1
                v31 = v3 - v1

                v21_norm = np.linalg.norm(v21)
                v31_norm = np.linalg.norm(v31)

                phi_1 = phi_vertex[v1_index]
                phi_2 = phi_vertex[v2_index]
                phi_3 = phi_vertex[v3_index]

                alpha = np.arccos(np.dot(v21, v31) / (v21_norm * v31_norm))

                a = (phi_2 - phi_1) / v21_norm
                b = (phi_3 - phi_1) / (v31_norm * np.sin(alpha)) - (phi_2 - phi_1) / (
                    v21_norm * np.tan(alpha)
                )

                # eta = v21 / v21_norm
                # tau = np.cross(eps, eta)

                E_eps = -dphi_centers[i]
                E_eta = -a
                E_tau = -b

                F = (1 / ep_hat) * E_eps * E_eps + E_eta * E_eta + E_tau * E_tau
                F *= (
                    -0.5
                    * (self.ep_ex - self.ep_in)
                    * self.mesh.normals[i]
                    * self.mesh.volumes[i]
                )

                f_db += factor_units * F

        # Ionic boundary force
        f_ib = (
            -0.5
            * factor_units
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

    def calculate_solvation_forces(
        self,
        h=0.001,
        force_formulation="maxwell_tensor",
        fdb_approx=False,
        units="kcal_molA",
    ):
        """Calculate total electrostatic solvation forces acting on the solute.
        Based on https://doi.org/10.1021/acs.jctc.3c00021
        (change units conversion to kcal/mol/Å)

        Computes the forces using either a boundary-integral energy functional approach
        or an integration of the Maxwell stress tensor over the molecular surface mesh.
        Calculations are scaled to unit conversions of kcal/mol/Å.

        Args:
            h (float, optional): Finite difference step size used to calculate numerical
                                 gradients if required. Defaults to 0.001.
            force_formulation (str, optional): Theoretical framework for force calculation.
                                               Options are "maxwell_tensor" or
                                               "energy_functional". Defaults to "maxwell_tensor".
            fdb_approx (bool, optional): If True, applies an normal-approximation to the dielectric
                                         boundary force component when using the energy
                                         functional formulation. Defaults to False.

        Raises:
            ValueError: If `force_formulation` is not one of the two supported strings.

        Side Effects:
            - Updates the `self.results` dictionary with calculated values
            - Updates execution profiling timestamps inside `self.timings`.
            - Prints processing time information to standard output if `self.print_times` is True.
        """
        if "phi" not in self.results:
            print(
                "Please compute surface potential first with simulation.calculate_potentials()"
            )
            return

        if force_formulation == "energy_functional":
            if "f_qf" not in self.results:
                self.calculate_gradient_field(h=h)
                self.calculate_charges_forces(units=units)

            self.calculate_boundary_forces(fdb_approx=fdb_approx, units=units)

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
                    " formulation",
                )

        elif force_formulation == "maxwell_tensor":

            if "f_ib" not in self.results:
                self.calculate_boundary_forces()

            start_time = time.time()

            N_elements = self.mesh.number_of_elements
            P_normal = np.zeros([N_elements])
            phi_vertex = self.results["phi"].coefficients
            ep_hat = self.ep_in / self.ep_ex
            dphi_centers = (
                ep_hat * self.results["d_phi"].evaluate_on_element_centers()[0]
            )
            total_force = np.zeros(3)
            factor_units = convert_units("kcal_molA")

            for i in range(N_elements):
                eps = self.mesh.normals[i]

                # get vertex indices adyacent to a triangular element
                v1_index = self.mesh.elements[0, i]
                v2_index = self.mesh.elements[1, i]
                v3_index = self.mesh.elements[2, i]

                # get vertex coordinates from vertex indices
                v1 = self.mesh.vertices[:, v1_index]
                v2 = self.mesh.vertices[:, v2_index]
                v3 = self.mesh.vertices[:, v3_index]

                v21 = v2 - v1
                v31 = v3 - v1

                v21_norm = np.linalg.norm(v21)
                v31_norm = np.linalg.norm(v31)

                phi_1 = phi_vertex[v1_index]
                phi_2 = phi_vertex[v2_index]
                phi_3 = phi_vertex[v3_index]

                alpha = np.arccos(np.dot(v21, v31) / (v21_norm * v31_norm))

                a = (phi_2 - phi_1) / v21_norm
                b = (phi_3 - phi_1) / (v31_norm * np.sin(alpha)) - (phi_2 - phi_1) / (
                    v21_norm * np.tan(alpha)
                )

                eta = v21 / v21_norm
                tau = np.cross(eps, eta)

                E_eps = -dphi_centers[i]
                E_eta = -a
                E_tau = -b

                E_norm = np.sqrt(E_eps * E_eps + E_eta * E_eta + E_tau * E_tau)

                F = (
                    (E_eps * E_eps - 0.5 * E_norm * E_norm) * eps
                    + E_eps * E_eta * eta
                    + E_eps * E_tau * tau
                )

                F *= self.ep_ex
                total_force += F * self.mesh.volumes[i]
                P_normal[i] = np.sqrt(
                    np.dot(F * self.mesh.volumes[i], F * self.mesh.volumes[i])
                )

            self.results["P_normal"] = factor_units * P_normal
            self.results["f_solv"] = factor_units * total_force + self.results["f_ib"]
            self.timings["time_calc_solvation_force"] = time.time() - start_time
            if self.print_times:
                print(
                    "It took ",
                    self.timings["time_calc_solvation_force"],
                    " seconds to compute the solvation forces with ",
                    force_formulation,
                    " formulation",
                )

        else:
            raise ValueError(
                'Formulation have to be "maxwell_tensor" or "energy_functional"'
            )

    def calculate_coulomb_potential(self, eval_points):
        r"""Calculate the vacuum Coulomb potential at a set of evaluation points.

        Computes the primary electrostatic potential generated by all explicit point
        charges within the solute molecule.

        $$\phi_{\text{coul}}(\mathbf{r}) = \sum_{i} \frac{q_i}{4\pi \|\mathbf{r} - \mathbf{r}_i\|}$$

        Args:
            eval_points (array_like): An $N \times 3$ array or matrix of Cartesian coordinates
                                      representing the target points where the potential
                                      is evaluated.

        Returns:
            np.ndarray: A 1D array of length $N$ containing the cumulative electrostatic
                        Coulomb potential at each evaluation point.
        """
        phi_coul = np.zeros(len(eval_points), dtype=float)

        for i in range(len(self.x_q)):
            dist = np.linalg.norm(eval_points - self.x_q[i, :], axis=1)
            phi_coul[:] += self.q[i] / (4 * np.pi * dist[:])

        return phi_coul

    def create_sas_mesh(self, sas_mesh_density=None):
        """Generate the Solvent Accessible Surface (SAS) mesh for the solute.

        This method builds a SAS mesh by expanding the physical radii of the solute's
        atoms by the `mesh_probe_radius`. It writes a temporary `.pqr` and `.xyzr` file,
        runs the chosen external mesh generator (MSMS or NanoShaper) with a tiny secondary
        probe radius (0.05), imports the resulting triangular mesh grid, and clean up
        files if requested.

        Args:
            sas_mesh_density (float, optional): Density of the generated surface mesh.
                                                If provided, overrides `self.sas_mesh_density`.


        Side Effects:
            - Updates `self.sas_mesh_density` if an explicit density argument is passed.
            - Populates `self.sas_mesh` with the imported surface grid data.
            - May create and modify a directory containing intermediate structural and mesh files.
        """
        if sas_mesh_density is not None:
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

        probe_radius = 0.05  # small probe for SAS
        if self.mesh_generator == "msms":
            mesh_tools.generate_msms_mesh(
                sas_mesh_xyzr_file,
                sas_mesh_dir,
                self.solute_name + "_sas",
                self.sas_mesh_density,
                probe_radius,
            )

        if self.mesh_generator == "nanoshaper":
            nanoshaper_grid_scale = (
                mesh_tools.density_to_nanoshaper_grid_scale_conversion(
                    self.sas_mesh_density
                )
            )
            mesh_tools.generate_nanoshaper_mesh(
                sas_mesh_xyzr_file,
                sas_mesh_dir,
                self.solute_name + "_sas",
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
    """Extract the solute name from the first line of a PDB file.

    Reads the header/initial line of a specified PDB file, splits the line
    by blocks of two or more consecutive whitespace characters, and extracts
    the fourth element (index 3) as the lowercase identifier for the solute.

    Args:
        pdb_path (str): The file system path to the target PDB file.

    Returns:
        str: The extracted name of the solute in lowercase.
    """
    pdb_file = open(pdb_path)
    first_line = pdb_file.readline()
    first_line_split = re.split(r"\s{2,}", first_line)
    solute_name = first_line_split[3].lower()
    pdb_file.close()

    return solute_name


def convert_units(units):
    """Computes the scalar conversion factor from standard atomic units ($\text{e}/\varepsilon_0\text{Å}$) to target units.

    Supports conversion into SI millivolts, thermal voltage equivalents, or thermodynamic energy units per charge.

    Args:
        units (str): The desired output unit key identifier. Acceptable values include
            'mV', 'kT_e', 'kJ_mol_e', 'kJ_mol', 'kJ_molA', 'kcal_mol_e', 'kcal_mol', 'kcal_molA', and 'e_eps0_angs'.

    Returns:
        float: Multiplicative scaling coefficient to transform the raw electrostatic potential value.
    """
    units = str(units).strip().lower().replace("-", "_").replace(" ", "_")
    units = units.replace("__", "_")

    qe = 1.60217663e-19
    eps0 = 8.8541878128e-12
    ang_to_m = 1e-10
    kb = 1.380649e-23
    kT = kb * 298.15  # Assuming temperature of 298.15 K
    Na = 6.02214076e23

    to_V = qe / (eps0 * ang_to_m)

    if units == "mv":
        unit_conversion = to_V * 1000
    elif units == "kt_e":
        unit_conversion = to_V / (kT / qe)
    elif units in ["kj_mol_e", "kj_mol", "kj_molA"]:
        unit_conversion = to_V * (qe * Na / 1000)
    elif units in ["kcal_mol_e", "kcal_mol", "kcal_mola"]:
        unit_conversion = to_V * (qe * Na / (4.184 * 1000))
    elif units == "e_eps0_angs":
        unit_conversion = 1.0
    else:
        print("Units not recognized. Defaulting to mV")
        unit_conversion = to_V * 1000

    return unit_conversion
