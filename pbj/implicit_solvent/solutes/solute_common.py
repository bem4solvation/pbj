import re
import bempp_cl as bempp
import bempp_cl.api
import os
import numpy as np
import time
import shutil
import pbj.mesh.mesh_tools as mesh_tools
import pbj.mesh.charge_tools as charge_tools
import pbj.implicit_solvent.pb_formulation as pb_formulations
import pbj.implicit_solvent.utils as utils


class Solute:
    """The basic Solute object
    This object holds all the solute information and allows for an easy way to hold the data
    """

    def __init__(
        self,
        solute_file_path,
        save_mesh_build_files=False,
        mesh_build_files_dir="mesh_files/",
        mesh_density=2.0,
        nanoshaper_grid_scale=None,
        solvent_radius=1.4,
        mesh_generator="nanoshaper",
        print_times=False,
        solute_type="lpbe",
        formulation="direct",
        fill_cavities=True,
        cavity_cutoff=60,
    ):
        """Initialize a solute object, configure solver parameters, and load mesh/charge data.

        The constructor imports the requested structure file, generates or loads a surface mesh,
        and initializes the electrostatic formulation, radii, and cavity settings used later
        by the Poisson-Boltzmann workflow.

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
            formulation (str, optional): Electrostatic formulation type used by the PB solver.
                Common choices include "direct", "direct_external", "direct_permuted",
                "direct_external_permuted", "direct_stern", "direct_amoeba",
                "alpha_beta", "alpha_beta_external_potential",
                "alpha_beta_single_blocked", "first_kind_external",
                "first_kind_internal", "juffer", "lu", "muller_external",
                "muller_internal", "slic", and "slic_prop". Use
                display_available_formulations() to print the full set of supported
                formulations at runtime. Defaults to "direct".
            radius_keyword (str, optional): Keyword used to select atom radii from the input data.
                Defaults to "solute".
            solute_radius_type (str, optional): Target classification for the solute radii.
                Defaults to "PB".
            fill_cavities (bool, optional): If True, include cavity-filling behavior during mesh setup.
                Defaults to True.
            cavity_cutoff (int, optional): Cutoff value used in cavity-filling logic.
                Defaults to 60.

        Raises:
            ValueError: If the specified formulation does not match any available
                module in pb_formulations.
        """

        if not os.path.isfile(solute_file_path):
            print("file does not exist -> Cannot start")
            return

        self.save_mesh_build_files = save_mesh_build_files
        self.mesh_build_files_dir = os.path.abspath(mesh_build_files_dir)

        self._pb_formulation = formulation
        self.solute_type = solute_type

        solute_formulation_module = getattr(pb_formulations, self.solute_type, None)
        self.formulation_object = getattr(
            solute_formulation_module, self._pb_formulation, None
        )
        if self.formulation_object is None:
            raise AttributeError(
                f"Formulation '{self._pb_formulation}' not found inside pb_formulations.{self.solute_type}"
            )

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
        self.fill_cavities = fill_cavities
        self.cavity_cutoff = cavity_cutoff

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

    @property
    def pb_formulation(self):
        return self._pb_formulation

    @pb_formulation.setter
    def pb_formulation(self, value):
        self._pb_formulation = value

        solute_formulation_module = getattr(pb_formulations, self.solute_type, None)
        self.formulation_object = getattr(
            solute_formulation_module, self._pb_formulation, None
        )
        if (
            "preconditioning_matrix_gmres" not in self.matrices
        ):  # might already exist if just regenerating RHS
            self.matrices["preconditioning_matrix_gmres"] = None
        if self.formulation_object is None:
            raise ValueError("Unrecognised formulation type %s" % self.pb_formulation)

    def initialise_matrices(self):
        start_time = time.time()  # Start the timing for the matrix construction
        # Construct matrices based on the desired formulation
        # Verify if parameters are already set and save A matrix
        if self.formulation_object.verify_parameters(self):
            self.formulation_object.lhs(self)
        self.timings["time_matrix_initialisation"] = time.time() - start_time

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

    def get_surface_potential_derivative(
        self, units="kcal_molA", print_units=True, internal_derivative=False
    ):
        r"""Return the surface-potential derivative values from the latest solve.

        Retrieves the stored derivative data for the surface potential and applies an
        optional scaling factor based on the interior/exterior permittivity ratio when
        `internal_derivative` is requested.

        Args:
            units (str, optional): Unit identifier passed to `convert_units` for the
                returned values. Defaults to "kcal_molA".
            print_units (bool, optional): If True, prints the unit label to standard output.
                Defaults to True.
            internal_derivative (bool, optional): If True, scales the returned derivative by
                the ratio `self.ep_in / self.ep_ex`. Defaults to False.

        Returns:
            tuple: A tuple containing the derivative coefficients and the corresponding
                unit label.

        Side Effects:
            - Prints the unit label if `print_units` is True.
        """

        if "phi" not in self.results:
            print(
                "Please compute surface potential first with simulation.calculate_potentials()"
            )
            return
        factor_ep = 1
        if internal_derivative:
            factor_ep = self.ep_in / self.ep_ex

        unit_conversion, unit_label = charge_tools.convert_units(
            units, magnitude="d_potential"
        )
        if print_units:
            print(f"Units {unit_label} for potential")

        return (
            self.results["d_phi"].coefficients * unit_conversion * factor_ep,
            unit_label,
        )

    def get_surface_potential(self, units="kcal_mol", print_units=True):
        r"""Return the surface-potential values from the latest solve.

        Retrieves the stored surface-potential coefficients from the previous boundary-element
        solve and converts them to the requested units.

        Args:
            units (str, optional): Unit identifier passed to `convert_units` for the
                returned values. Defaults to "kcal_mol".
            print_units (bool, optional): If True, prints the unit label to standard output.
                Defaults to True.

        Returns:
            tuple: A tuple containing the potential coefficients and the corresponding
                unit label.

        Side Effects:
            - Prints the unit label if `print_units` is True.
        """
        if "d_phi" not in self.results:
            print(
                "Please compute surface potential first with simulation.calculate_surface_potential()"
            )
            return
        unit_conversion, unit_label = charge_tools.convert_units(
            units, magnitude="potential"
        )
        if print_units:
            print(f"Units {unit_label} for potential")
        return self.results["phi"].coefficients * unit_conversion, unit_label


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
