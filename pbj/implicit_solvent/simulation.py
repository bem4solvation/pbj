import bempp_cl as bempp
import bempp_cl.api
import time
import trimesh
import pbj.mesh.plotting_tools as plotting_tools
import pbj.mesh.charge_tools as charge_tools
import numpy as np
import pbj.implicit_solvent.solutes as pb_solutes
import pbj.implicit_solvent.pb_formulation as pb_formulations


class Simulation:
    """Manages the boundary element method (BEM) simulation environment for implicit solvent models.

    Coordinates one or more solute molecules, sets up the Poisson-Boltzmann (PB) formulation,
    assembles the global linear systems, applies preconditioning, and executes boundary element
    solvers to calculate electrostatic potentials, solvation energies, forces, and effective
    near-surface (ENS) potentials.
    """

    def __init__(
        self,
        simulation_type="lpbe",
        force_field="amber",
        formulation="direct",
        stern_layer=False,
        print_times=False,
    ):
        """Initializes a Simulation environment with a specific Poisson-Boltzmann formulation.

        Args:
            formulation (str, optional): The PB boundary integral formulation to use
                (e.g., 'direct', 'direct_stern', 'slic', 'direct_amoeba'). Defaults to "direct".
            simulation_type (str, optional): The type of simulation to run (e.g., 'lpbe', 'npbe'). Defaults to 'lpbe'.
            stern_layer (bool, optional): If True, incorporates a Stern (ion-exclusion) layer
                into the formulation workspace. Defaults to False.
            print_times (bool, optional): If True, displays execution times during major solver
                phases. Defaults to False.
        """

        if stern_layer and formulation != "slic":
            self._pb_formulation = "direct_stern"
            if formulation not in ("direct", "direct_stern"):
                print(
                    "Stern or ion-exclusion layer only supported with direct formulation. Using direct."
                )
        else:
            self._pb_formulation = formulation

        if formulation in ("direct_stern", "slic"):
            stern_layer = True

        if force_field == "amoeba" and simulation_type == "lpbe":
            self.solute_type = "lpbe_amoeba"
        else:
            self.solute_type = simulation_type

        self.solute_object = getattr(pb_solutes, self.solute_type, None)
        if self.solute_object is None:
            raise ValueError("Unrecognised solute type %s" % self.solute_type)

        solute_formulation_module = getattr(pb_formulations, self.solute_type, None)
        if solute_formulation_module is None:
            raise ValueError(
                f"Unrecognised formulation module '{self.solute_type}' in pb_formulations"
            )

        self.formulation_object = getattr(
            solute_formulation_module, self._pb_formulation, None
        )
        if self.formulation_object is None:
            raise AttributeError(
                f"Formulation '{self._pb_formulation}' not found inside pb_formulations.{self.solute_type}"
            )

        self.gmres_tolerance = 1e-5
        self.gmres_restart = 1000
        self.gmres_max_iterations = 1000

        self.induced_dipole_iter_tol = 1e-2  # AMOEBA?

        self.slic_max_iterations = 20
        self.slic_tolerance = 1e-4

        self.solutes = list()
        self.solutes_names = list()
        self.matrices = dict()
        self.rhs = dict()
        self.timings = dict()
        self.run_info = dict()

        self.ep_ex = 80.0
        self.kappa = 0.125

        self.pb_formulation_preconditioning = True

        if (
            self._pb_formulation == "direct"
            or self._pb_formulation == "direct_stern"
            or self._pb_formulation == "slic"
            or self._pb_formulation == "direct_amoeba"
        ):
            self.pb_formulation_preconditioning_type = "block_diagonal"
        else:
            self.pb_formulation_preconditioning_type = "mass_matrix"

        self.operator_assembler = "dense"

        self.SOR = 0.7

    @property
    def pb_formulation(self):
        return self._pb_formulation

    @pb_formulation.setter
    def pb_formulation(self, value):
        self._pb_formulation = value

        solute_formulation_module = getattr(pb_formulations, self.solute_type, None)
        if solute_formulation_module is None:
            raise ValueError(
                f"Unrecognised formulation module '{self.solute_type}' in pb_formulations"
            )

        self.formulation_object = getattr(
            solute_formulation_module, self._pb_formulation, None
        )
        if self.formulation_object is None:
            raise AttributeError(
                f"Formulation '{self._pb_formulation}' not found inside pb_formulations.{self.solute_type}"
            )
        self.matrices["preconditioning_matrix_gmres"] = None
        # reset solute
        if len(self.solutes) > 0:
            for index, solute in enumerate(self.solutes):
                solute.pb_formulation = self.pb_formulation

    @property
    def pb_formulation_preconditioning(self):
        return self._pb_formulation_preconditioning

    @pb_formulation_preconditioning.setter
    def pb_formulation_preconditioning(self, value):
        self._pb_formulation_preconditioning = value
        # reset solute
        if len(self.solutes) > 0:
            for index, solute in enumerate(self.solutes):
                solute.pb_formulation_preconditioning = (
                    self.pb_formulation_preconditioning
                )

    @property
    def pb_formulation_preconditioning_type(self):
        return self._pb_formulation_preconditioning_type

    @pb_formulation_preconditioning_type.setter
    def pb_formulation_preconditioning_type(self, value):
        self._pb_formulation_preconditioning_type = value
        # reset solute
        if len(self.solutes) > 0:
            for index, solute in enumerate(self.solutes):
                solute.pb_formulation_preconditioning_type = (
                    self.pb_formulation_preconditioning_type
                )

    @property
    def ep_ex(self):
        return self._ep_ex

    @ep_ex.setter
    def ep_ex(self, value):
        self._ep_ex = value
        # reset solute
        if len(self.solutes) > 0:
            for index, solute in enumerate(self.solutes):
                solute.ep_ex = self.ep_ex
                solute.e_hat_stern = solute.ep_stern / solute.ep_ex
                solute.pb_formulation_beta = solute.ep_ex / solute.ep_in  # np.nan

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, value):
        self._kappa = value
        # reset solute
        if len(self.solutes) > 0:
            for index, solute in enumerate(self.solutes):
                solute.kappa = self.kappa

    def add_solute(self, solute, name=None):
        """Register a solute molecule in the simulation context and synchronize parameters.

        Validates that the provided object conforms to the expected `Solute` specification,
        then propagates simulation-wide global variables (for example, external permittivity,
        ionic strength parameter, and preconditioning settings) down to the solute object.

        Args:
            solute (Solute): An instance of the `pbj.implicit_solvent.solute.Solute` class
                to be modeled.
            name (str, optional): Display name to associate with the solute in the simulation.
                If not provided, the solute's own `solute_name` is used. Defaults to None.

        Raises:
            ValueError: If the input object is not an instance of the `Solute` class or lacks
                proper structural initialization.
        """

        if solute.solute_type == self.solute_type and hasattr(solute, "solute_name"):
            if solute in self.solutes:
                print(
                    "Solute object is already added to this simulation. Ignoring this add command."
                )
            # elif solute.solute_type != self.solute_type:
            #     raise ValueError(
            #         f"Solute type '{solute.solute_type}' does not match simulation solute type '{self.solute_type}'."
            #     )
            else:
                solute.ep_ex = self.ep_ex
                solute.kappa = self.kappa
                solute.SOR = self.SOR
                solute.induced_dipole_iter_tol = self.induced_dipole_iter_tol
                solute.operator_assembler = self.operator_assembler
                solute.pb_formulation_preconditioning = (
                    self.pb_formulation_preconditioning
                )
                solute.pb_formulation_preconditioning_type = (
                    self.pb_formulation_preconditioning_type
                )
                if (
                    self.pb_formulation[-5:] == "stern" or self.pb_formulation == "slic"
                ):  # Think of better way to do this
                    solute.stern_mesh_density = (
                        solute.stern_mesh_density_ratio * solute.sas_mesh_density
                    )
                # if solute.force_field == "amoeba":
                #     if self.pb_formulation not in ("direct", "direct_amoeba"):
                #         print(
                #             "AMOEBA force field is only supported for direct formulation with no Stern layer. Using direct"
                #         )
                #     self.pb_formulation = "direct_amoeba"
                self.solutes.append(solute)
                if isinstance(name, str):
                    self.solutes_names.append(name)
                else:
                    self.solutes_names.append(solute.solute_name)
        else:
            raise ValueError(
                "Given object is not of the 'Solute' class of the simulation or pdb/pqr file not correctly loaded."
            )

    def create_and_assemble_linear_system(self):
        """Builds and assembles the global blocked discrete operator matrix and right-hand side vectors.

        Iterates through all registered solutes to compute self-interaction matrices, cross-interaction
        coupling operators between distinct solute groups, global boundary condition right-hand side
        vectors, and a block-diagonal or mass-matrix preconditioning operator for the global GMRES system.

        Returns:
            None
        """
        from scipy.sparse import bmat, dok_matrix
        from scipy.sparse.linalg import aslinearoperator

        solute_count = len(self.solutes)
        # A = bempp.api.BlockedDiscreteOperator(solute_count * 2, solute_count * 2)

        A = np.empty((solute_count, solute_count), dtype="O")

        precond_matrix = []

        rhs_final_discrete = []

        # Get self interactions of each solute
        for index, solute in enumerate(self.solutes):
            solute.pb_formulation = self.pb_formulation

            solute.initialise_matrices()
            solute.initialise_rhs()
            solute.apply_preconditioning()

            # A[index * 2, index * 2] = solute.matrices["A_discrete"][0, 0]
            # A[(index * 2) + 1, index * 2] = solute.matrices["A_discrete"][1, 0]
            # A[index * 2, (index * 2) + 1] = solute.matrices["A_discrete"][0, 1]
            # A[(index * 2) + 1, (index * 2) + 1] = solute.matrices["A_discrete"][1, 1]

            A[index, index] = solute.matrices["A_discrete"]

            self.rhs["rhs_" + str(index + 1)] = [
                solute.rhs["rhs_1"],
                solute.rhs["rhs_2"],
            ]

            rhs_final_discrete.extend(solute.rhs["rhs_discrete"])

            if solute.matrices["preconditioning_matrix_gmres"] is not None:

                if solute.stern_object is None:
                    precond_matrix_top_row = []
                    precond_matrix_bottom_row = []

                    for index_source, solute_source in enumerate(self.solutes):

                        if index_source == index:
                            precond_matrix_top_row.extend(
                                solute.matrices["preconditioning_matrix_gmres"][0]
                            )
                            precond_matrix_bottom_row.extend(
                                solute.matrices["preconditioning_matrix_gmres"][1]
                            )
                        else:
                            M = solute.dirichl_space.grid_dof_count
                            N = solute_source.dirichl_space.grid_dof_count
                            zero_matrix = dok_matrix((M, N))
                            precond_matrix_top_row.extend([zero_matrix, zero_matrix])
                            precond_matrix_bottom_row.extend([zero_matrix, zero_matrix])

                    precond_matrix.extend(
                        [precond_matrix_top_row, precond_matrix_bottom_row]
                    )

                else:
                    precond_matrix_row_0 = []
                    precond_matrix_row_1 = []
                    precond_matrix_row_2 = []
                    precond_matrix_row_3 = []

                    for index_source, solute_source in enumerate(self.solutes):

                        # if index_source == index:
                        if solute == solute_source:
                            precond_matrix_row_0.extend(
                                solute.matrices["preconditioning_matrix_gmres"][0]
                            )
                            precond_matrix_row_1.extend(
                                solute.matrices["preconditioning_matrix_gmres"][1]
                            )
                            precond_matrix_row_2.extend(
                                solute.matrices["preconditioning_matrix_gmres"][2]
                            )
                            precond_matrix_row_3.extend(
                                solute.matrices["preconditioning_matrix_gmres"][3]
                            )
                        else:
                            M_diel = solute.dirichl_space.grid_dof_count
                            N_diel = solute_source.dirichl_space.grid_dof_count
                            M_stern = solute.stern_object.dirichl_space.grid_dof_count
                            N_stern = (
                                solute_source.stern_object.dirichl_space.grid_dof_count
                            )

                            zero_matrix = dok_matrix((M_diel, N_diel))
                            precond_matrix_row_0.extend([zero_matrix, zero_matrix])
                            precond_matrix_row_1.extend([zero_matrix, zero_matrix])

                            zero_matrix = dok_matrix((M_stern, N_diel))
                            precond_matrix_row_2.extend([zero_matrix, zero_matrix])
                            precond_matrix_row_3.extend([zero_matrix, zero_matrix])

                            zero_matrix = dok_matrix((M_diel, N_stern))
                            precond_matrix_row_0.extend([zero_matrix, zero_matrix])
                            precond_matrix_row_1.extend([zero_matrix, zero_matrix])

                            zero_matrix = dok_matrix((M_stern, N_stern))
                            precond_matrix_row_2.extend([zero_matrix, zero_matrix])
                            precond_matrix_row_3.extend([zero_matrix, zero_matrix])

                    precond_matrix.extend(
                        [
                            precond_matrix_row_0,
                            precond_matrix_row_1,
                            precond_matrix_row_2,
                            precond_matrix_row_3,
                        ]
                    )

                    self.rhs["rhs_" + str(index + 1)].extend(
                        [solute.rhs["rhs_3"], solute.rhs["rhs_4"]]
                    )

        if len(precond_matrix) > 0:
            precond_matrix_full = bmat(precond_matrix).tocsr()
            self.matrices["preconditioning_matrix_gmres"] = aslinearoperator(
                precond_matrix_full
            )

        # Calculate matrix elements for interactions between solutes

        for index_target, solute_target in enumerate(self.solutes):
            i = index_target
            solute_target.matrices["A_inter"] = []
            for index_source, solute_source in enumerate(self.solutes):
                j = index_source

                if i != j:

                    self.formulation_object.lhs_inter_solute_interactions(
                        self, solute_target, solute_source
                    )
                    if i > j:
                        index_array = j
                    else:
                        index_array = j - 1

                    A[i, j] = solute_target.matrices["A_inter"][
                        index_array
                    ].weak_form()  # always weak form as it's not
                    # touched by preconditioner

        # self.matrices["A"] = A
        A_discrete = bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(A)
        self.matrices["A_discrete"] = A_discrete
        self.rhs["rhs_discrete"] = rhs_final_discrete

    def create_and_assemble_rhs(self):
        """Re-initializes and assembles the global right-hand side vectors for all solutes.

        Useful when atomic coordinates, charges, or boundary conditions have changed but the
        integral operator matrices remain invariant.

        Returns:
            None
        """
        # from scipy.sparse import bmat, dok_matrix
        # from scipy.sparse.linalg import aslinearoperator

        # solute_count = len(self.solutes)

        rhs_final_discrete = []

        for index, solute in enumerate(self.solutes):
            solute.pb_formulation = self.pb_formulation

            solute.initialise_rhs()
            solute.apply_preconditioning_rhs()

            self.rhs["rhs_" + str(index + 1)] = [
                solute.rhs["rhs_1"],
                solute.rhs["rhs_2"],
            ]

            rhs_final_discrete.extend(solute.rhs["rhs_discrete"])

        self.rhs["rhs_discrete"] = rhs_final_discrete

    def calculate_surface_potential(self, rerun_all=False, rerun_rhs=False):
        r"""Solves the boundary element matrix equation to obtain the surface electrostatic potential.

        Invokes the underlying formulation object's calculation routine to execute the
        Krylov subspace solver (e.g., GMRES) and determine the Dirichlet ($\\phi$) and
        Neumann ($\\partial\\phi/\\partial n$) distributions across the molecular interface.

        Args:
            rerun_all (bool, optional): Forces a full re-assembly of the matrix operators
                and RHS vectors. Defaults to False.
            rerun_rhs (bool, optional): Forces a re-assembly of only the RHS vector before solving.
                Defaults to False.

        Returns:
            None
        """

        if len(self.solutes) == 0:
            print("Simulation has no solutes loaded")
        else:
            self.formulation_object.calculate_potential(self, rerun_all, rerun_rhs)

        # Print times, if this is desired

    # if self.print_times:
    #           show_potential_calculation_times(self)

    def calculate_solvation_energy(
        self,
        electrostatic_energy=True,
        nonpolar_energy=False,
        rerun_all=False,
        rerun_rhs=False,
        units="kcal_mol",
    ):
        """Computes the total solvation free energy for all registered solute molecules.

        Calculates the electrostatic contribution via boundary surface integrations and/or
        nonpolar cavitation/dispersion terms based on the solved interface state.

        Args:
            electrostatic_energy (bool, optional): If True, evaluates the polar/electrostatic
                component of solvation. Defaults to True.
            nonpolar_energy (bool, optional): If True, evaluates the nonpolar/cavitation
                component of solvation. Defaults to False.
            rerun_all (bool, optional): Triggers full operator re-assembly before potential calculation.
                Defaults to False.
            rerun_rhs (bool, optional): Triggers RHS re-assembly before potential calculation.
                Defaults to False.
            units (str, optional): Unit identifier passed to `convert_units` for the
                computed solvation energies. Defaults to "kcal_mol".

        Returns:
            None
        """
        if len(self.solutes) == 0:
            print("Simulation has no solutes loaded")
            return

        if rerun_all:
            self.calculate_surface_potential(rerun_all=rerun_all)

        if rerun_rhs:
            self.calculate_surface_potential(rerun_rhs=rerun_rhs)

        if "phi" not in self.solutes[0].results:
            # If surface potential has not been calculated, calculate it now
            self.calculate_surface_potential()

        start_time = time.time()
        for index, solute in enumerate(self.solutes):

            solute.calculate_solvation_energy(
                electrostatic_energy, nonpolar_energy, units=units
            )

        self.timings["time_calc_energy"] = time.time() - start_time

    def calculate_solvation_forces(
        self,
        h=0.001,
        rerun_all=False,
        rerun_rhs=False,
        force_formulation="maxwell_tensor",
        fdb_approx=False,
        units="kcal_molA",
    ):
        """Evaluates the solvation forces acting on each atom of the solute molecules.

        Computes the gradient of the solvation energy using boundary integral distributions,
        supporting methods such as the Maxwell stress tensor integration or direct boundary
        derivative approximations.

        Args:
            h (float, optional): Finite difference step size or regularization parameter. Defaults to 0.001.
            rerun_all (bool, optional): Triggers full re-computation of surface potentials if True.
                Defaults to False.
            force_formulation (str, optional): Formulation type for force evaluation, such as
                'maxwell_tensor' or 'energy_functional'. Defaults to "maxwell_tensor".
            fdb_approx (bool, optional): If True, applies an normal-approximation to the dielectric
                boundary force component when using the energy functional formulation.
            units (str, optional): Unit identifier passed to `convert_units` for the
                computed solvation forces. Defaults to "kcal_molA".

        Returns:
            None
        """

        if rerun_all:
            self.calculate_surface_potential(rerun_all=rerun_all)

        if rerun_rhs:
            self.calculate_surface_potential(rerun_rhs=rerun_rhs)

        if len(self.solutes) == 0:
            print("Simulation has no solutes loaded")
            return

        if "phi" not in self.solutes[0].results:
            # If surface potential has not been calculated, calculate it now
            self.calculate_surface_potential()

        start_time = time.time()
        for index, solute in enumerate(self.solutes):
            solute.calculate_solvation_forces(
                h=h,
                force_formulation=force_formulation,
                fdb_approx=fdb_approx,
                units=units,
            )

        self.timings["time_calc_force"] = time.time() - start_time

    def calculate_potential_solvent(
        self, eval_points, units="mV", rerun_all=False, rerun_rhs=False
    ):
        """Evaluates the electrostatic potential at specified points residing in the solvent region.

        Excludes and masks any points located inside the interior domain of any registered solute
        using a spatial containment check, then evaluates Laplace or modified Helmholtz single-
        and double-layer potential operators.

        Args:
            eval_points (numpy.ndarray): 2D array of shape $(N, 3)$ specifying the Cartesian coordinates
                of the $N$ evaluation target points.
            units (str, optional): Target conversion units for the potential output ('mV', 'kT_e',
                'kcal_mol_e', 'kJ_mol_e', 'e_eps0_angs'). Defaults to "mV".
            rerun_all (bool, optional): Forces full potential re-assembly if True. Defaults to False.
            rerun_rhs (bool, optional): Forces RHS re-assembly if True. Defaults to False.

        Returns:
            tuple: A tuple containing two elements:
                - phi_solvent (numpy.ndarray): 1D array of shape $(N,)$ representing the converted
                  electrostatic potentials at the valid points (points inside solutes are set to $0.0$).
                - points_solvent (numpy.ndarray): 1D boolean array of shape $(N,)$ marking True for
                  points belonging to the exterior solvent region.
        """

        if len(self.solutes) == 0:
            print("Simulation has no solutes loaded")
            return

        if rerun_all:
            self.calculate_surface_potential(rerun_all=rerun_all)

        if rerun_rhs:
            self.calculate_surface_potential(rerun_rhs=rerun_rhs)

        if "phi" not in self.solutes[0].results:
            # If surface potential has not been calculated, calculate it now
            self.calculate_surface_potential()

        eval_points = np.transpose(eval_points)  # format that Bempp likes

        # Mask out points in solute
        points_solvent = np.ones(np.shape(eval_points)[1], dtype=bool)
        for index, solute in enumerate(self.solutes):

            # Check if evaluation points are inside a solute
            verts = np.transpose(solute.mesh.vertices)
            faces = np.transpose(solute.mesh.elements)

            mesh_tri = trimesh.Trimesh(vertices=verts, faces=faces)

            points_solute = mesh_tri.contains(np.transpose(eval_points))

            points_solvent = np.logical_and(
                points_solvent, np.logical_not(points_solute)
            )

        # Compute potential
        phi_solvent = np.zeros(np.shape(eval_points)[1], dtype=float)
        for index, solute in enumerate(self.solutes):

            if self.kappa < 1e-12:
                V = bempp.api.operators.potential.laplace.single_layer(
                    solute.neumann_space, eval_points[:, points_solvent]
                )
                K = bempp.api.operators.potential.laplace.double_layer(
                    solute.dirichl_space, eval_points[:, points_solvent]
                )
            else:
                V = bempp.api.operators.potential.modified_helmholtz.single_layer(
                    solute.neumann_space, eval_points[:, points_solvent], self.kappa
                )
                K = bempp.api.operators.potential.modified_helmholtz.double_layer(
                    solute.dirichl_space, eval_points[:, points_solvent], self.kappa
                )

            phi_aux = (
                K * solute.results["phi"]
                - solute.ep_in / solute.ep_ex * V * solute.results["d_phi"]
            )
            phi_solvent[points_solvent] += phi_aux[0, :]

        unit_conversion, _ = charge_tools.convert_units(units)

        return unit_conversion * phi_solvent, points_solvent

    def calculate_reaction_potential_solute(
        self,
        eval_points,
        units="mV",
        solute_subset=None,
        rerun_all=False,
        rerun_rhs=False,
    ):
        """Evaluates the reaction electrostatic potential at specified points inside the solute domain.

        Computes the reaction component (the potential induced by the polarized solvent environment)
        by projecting boundary distributions back onto the internal evaluation coordinates.

        Args:
            eval_points (numpy.ndarray): 2D array of shape $(N, 3)$ specifying the Cartesian coordinates
                of the $N$ evaluation target points.
            units (str, optional): Target conversion units for the potential output ('mV', 'kT_e',
                'kcal_mol_e', 'kJ_mol_e', 'e_eps0_angs'). Defaults to "mV".
            solute_subset (numpy.ndarray or list, optional): Sub-indices of specific solutes to compute.
                If None, evaluates all solutes. Defaults to None.
            rerun_all (bool, optional): Forces full potential re-assembly if True. Defaults to False.
            rerun_rhs (bool, optional): Forces RHS re-assembly if True. Defaults to False.

        Returns:
            tuple: A tuple containing two elements:
                - phi_solute (numpy.ndarray): 1D array of shape $(N,)$ representing the reaction
                  potentials at the target coordinates.
                - points_solute (numpy.ndarray): 1D integer array of shape $(N,)$ indicating the index of
                  the containing solute for each point (or $-1$ if the point resides in the solvent).
        """

        if len(self.solutes) == 0:
            print("Simulation has no solutes loaded")
            return

        if rerun_all:
            self.calculate_surface_potential(rerun_all=rerun_all)

        if rerun_rhs:
            self.calculate_surface_potential(rerun_rhs=rerun_rhs)

        if "phi" not in self.solutes[0].results:
            # If surface potential has not been calculated, calculate it now
            self.calculate_surface_potential()

        if solute_subset is None:
            solute_subset = np.arange(len(self.solutes))

        eval_points = np.transpose(eval_points)  # format that Bempp likes

        # Mask out points in solute
        points_solute = -np.ones(np.shape(eval_points)[1])
        phi_solute = np.zeros(np.shape(eval_points)[1], dtype=float)

        for index in solute_subset:

            solute = self.solutes[index]

            # Check if evaluation points are inside a solute
            verts = np.transpose(solute.mesh.vertices)
            faces = np.transpose(solute.mesh.elements)

            mesh_tri = trimesh.Trimesh(vertices=verts, faces=faces)

            points_solute_local = mesh_tri.contains(np.transpose(eval_points))

            points_solute[points_solute_local] = index

            slp = bempp.api.operators.potential.laplace.single_layer(
                solute.neumann_space, eval_points[:, points_solute_local]
            )
            dlp = bempp.api.operators.potential.laplace.double_layer(
                solute.dirichl_space, eval_points[:, points_solute_local]
            )

            phi_aux = slp * solute.results["d_phi"] - dlp * solute.results["phi"]

            phi_solute[points_solute_local] = phi_aux[0, :]

        unit_conversion, _ = charge_tools.convert_units(units)

        return unit_conversion * phi_solute, points_solute

    def calculate_coulomb_potential_solute(
        self,
        eval_points,
        units="mV",
        solute_subset=None,
        rerun_all=False,
        rerun_rhs=False,
    ):
        """Evaluates the vacuum (Coulomb) electrostatic potential at specified points inside the solute domain.

        Calculates the direct potential generated by permanent atomic source charges in a homogeneous vacuum
        environment.

        Args:
            eval_points (numpy.ndarray): 2D array of shape $(N, 3)$ specifying the Cartesian coordinates
                of the $N$ evaluation target points.
            units (str, optional): Target conversion units for the potential output ('mV', 'kT_e',
                'kcal_mol_e', 'kJ_mol_e', 'e_eps0_angs'). Defaults to "mV".
            solute_subset (numpy.ndarray or list, optional): Sub-indices of specific solutes to compute.
                If None, evaluates all solutes. Defaults to None.
            rerun_all (bool, optional): Retained for signature consistency. Defaults to False.
            rerun_rhs (bool, optional): Retained for signature consistency. Defaults to False.

        Returns:
            tuple: A tuple containing two elements:
                - phi_coul_solute (numpy.ndarray): 1D array of shape $(N,)$ with the vacuum Coulomb potentials.
                - points_solute (numpy.ndarray): 1D integer array of shape $(N,)$ mapping each point
                  to its parent solute index (or $-1$ if in the solvent region).
        """

        if len(self.solutes) == 0:
            print("Simulation has no solutes loaded")
            return

        if solute_subset is None:
            solute_subset = np.arange(len(self.solutes))

        # Mask out points in solute
        points_solute = -np.ones(np.shape(eval_points)[0])
        phi_coul_solute = np.zeros(np.shape(eval_points)[0], dtype=float)

        for index in solute_subset:

            solute = self.solutes[index]

            # Check if evaluation points are inside a solute
            verts = np.transpose(solute.mesh.vertices)
            faces = np.transpose(solute.mesh.elements)

            mesh_tri = trimesh.Trimesh(vertices=verts, faces=faces)

            points_solute_local = mesh_tri.contains(eval_points)

            points_solute[points_solute_local] = index

            phi_coul_solute[points_solute_local] = solute.calculate_coulomb_potential(
                eval_points[points_solute_local]
            )

        unit_conversion, _ = charge_tools.convert_units(units)

        return unit_conversion * phi_coul_solute, points_solute

    def calculate_potential_ens(
        self,
        atom_name=["H"],
        mesh_dx=1.0,
        mesh_length=40.0,
        ion_radius_explode=3.5,
        rerun_all=False,
        rerun_rhs=False,
    ):
        """Calculates the Effective Near Surface (ENS) electrostatic potential for target atom types.

        Implements spatial numerical integrations over local grids to evaluate ion accessibility
        and effective screening potential fields adjacent to the molecular boundaries.

        Args:
            atom_name (list of str, optional): Atom identifiers from the topology map (e.g., `['H']`)
                for which the ENS potential will be computed. Defaults to ["H"].
            mesh_dx (float, optional): Grid cell spacing/increment size used for spatial numerical
                integration. Defaults to 1.0.
            mesh_length (float, optional): Linear cross-sectional dimension of the local bounding
                integration mesh. Defaults to 40.0.
            ion_radius_explode (float, optional): Inflated offset distance appended to atomic radii
                to establish steric ion-exclusion zones. Defaults to 3.5.
            rerun_all (bool, optional): Triggers full operator re-assembly before calculation if True.
                Defaults to False.
            rerun_rhs (bool, optional): Triggers RHS re-assembly before calculation if True.
                Defaults to False.

        Returns:
            None
        """

        if len(self.solutes) == 0:
            print("Simulation has no solutes loaded")
            return

        if rerun_all:
            self.calculate_surface_potential(rerun_all=rerun_all)

        if rerun_rhs:
            self.calculate_surface_potential(rerun_rhs=rerun_rhs)

        if "phi" not in self.solutes[0].results:
            # If surface potential has not been calculated, calculate it now
            self.calculate_surface_potential()

        qe = 1.60217663e-19
        eps0 = 8.8541878128e-12
        ang_to_m = 1e-10
        to_V = qe / (eps0 * ang_to_m)
        kT = 4.11e-21
        # Na = 6.02214076e23

        # check if atom_name is a single string
        if isinstance(atom_name, str):
            atom_name = [atom_name]

        r_explode = np.array([])
        x_q = np.empty((0, 3))
        for index, solute in enumerate(self.solutes):
            r_explode = np.append(r_explode, solute.r_q + ion_radius_explode)
            x_q = np.append(x_q, solute.x_q, axis=0)

        for index, solute in enumerate(self.solutes):

            H_atoms = []
            for i, name in enumerate(solute.atom_name):
                if name in atom_name:
                    H_atoms.append(i)

            phi_ens = np.zeros(len(H_atoms))

            for i in range(len(H_atoms)):

                N = int(mesh_length / mesh_dx)
                ctr = solute.x_q[H_atoms[i]]
                x = np.linspace(
                    ctr[0] - mesh_length / 2, ctr[0] + mesh_length / 2, num=N
                )
                y = np.linspace(
                    ctr[1] - mesh_length / 2, ctr[1] + mesh_length / 2, num=N
                )
                z = np.linspace(
                    ctr[2] - mesh_length / 2, ctr[2] + mesh_length / 2, num=N
                )

                X, Y, Z = np.meshgrid(x, y, z)
                pos_mesh = np.zeros((3, N * N * N))
                pos_mesh[0, :] = X.flatten()
                pos_mesh[1, :] = Y.flatten()
                pos_mesh[2, :] = Z.flatten()

                inside = np.zeros(len(pos_mesh[0]), dtype=int)

                for j in range(len(r_explode)):
                    r_q_mesh = np.linalg.norm(x_q[j, :] - pos_mesh.transpose(), axis=1)
                    inside_local_index = np.nonzero(r_q_mesh < r_explode[j])[0]
                    inside[inside_local_index] += 1

                # r_mesh = np.linalg.norm(ctr - pos_mesh.transpose(), axis=1)

                outside = np.nonzero(inside == 0)[0]

                pos_mesh_outside = pos_mesh[:, outside]

                phi_solvent = np.zeros(len(outside))
                for index_source, solute_source in enumerate(self.solutes):

                    V = bempp_cl.api.operators.potential.modified_helmholtz.single_layer(
                        solute_source.neumann_space,
                        pos_mesh_outside,
                        self.kappa,
                        assembler=self.operator_assembler,
                    )
                    K = bempp_cl.api.operators.potential.modified_helmholtz.double_layer(
                        solute_source.dirichl_space,
                        pos_mesh_outside,
                        self.kappa,
                        assembler=self.operator_assembler,
                    )

                    phi_aux = (
                        K * solute_source.results["phi"]
                        - solute_source.ep_in
                        / solute_source.ep_ex
                        * V
                        * solute_source.results["d_phi"]
                    )

                    phi_solvent[:] += phi_aux[0, :]

                phi_V = phi_solvent * to_V

                pos_atom = solute.x_q[H_atoms[i]]
                dist = np.linalg.norm(pos_mesh_outside.transpose() - pos_atom, axis=1)
                G2_over_G2 = np.sum(np.exp(-qe * phi_V / kT) / dist**6) / np.sum(
                    np.exp(qe * phi_V / kT) / dist**6
                )

                phi_ens[i] = -kT / (2 * qe) * np.log(G2_over_G2) * 1000  # in mV
                bempp.api.log(
                    "PBJ: ENS calculation "
                    + str(atom_name)
                    + " atom %i, phi_ens = %1.3f mV" % (i, phi_ens[i])
                )

            solute.results["phi_ens"] = phi_ens

    def plot_surface_values(
        self,
        values="phi",
        units="kt",
        max_colorbar_scale=1,
        name="plot_surface.png",
        savefig=True,
        location="ses",
        show_axis=True,
        camera_view=None,
        isosurface_potential=None,
        min_max_vals=None,
        figsize=(900, 800),
        solutes_names=None,
        internal_derivative_plot=False,
    ):
        """Plot surface values for one or more registered solutes.

        The method prepares the requested surface quantity from each solute, optionally adds
        an isosurface overlay computed from solvent-region potentials, and delegates the actual
        rendering to the plotting helper.

        Args:
            values (str, optional): Surface quantity to display. Supported values are "phi"
                for the surface potential and "d_phi" for its derivative. Defaults to "phi".
            units (str, optional): Unit identifier passed to the solute accessors for
                value conversion. Defaults to "kt".
            max_colorbar_scale (float, optional): Scaling factor used to set the absolute
                colorbar range when `min_max_vals` is not given. Defaults to 1.
            name (str, optional): Output filename used when `savefig=True`. Defaults to
                "plot_surface.png".
            savefig (bool, optional): If True, writes the generated figure to disk.
                Defaults to True.
            location (str, optional): Surface mesh location to display. Use "ses" for the
                standard mesh or "stern" for the Stern-layer mesh when available.
                Defaults to "ses".
            show_axis (bool, optional): If True, shows the coordinate axes in the rendered scene.
                Defaults to True.
            camera_view (tuple, optional): Optional camera eye coordinates for the Plotly scene.
                Defaults to None.
            isosurface_potential (float, optional): If provided, computes and overlays an
                isosurface at this potential value. Defaults to None.
            min_max_vals (tuple, optional): Optional `(min_value, max_value)` pair used to
                override the automatic colorbar scale. Defaults to None.
            figsize (tuple, optional): Width and height of the figure in pixels.
                Defaults to `(900, 800)`.
            solutes_names (list, optional): List of solute indices or names to include.
                Defaults to None.
            internal_derivative_plot (bool, optional): If True, requests the internal
                derivative form when `values="d_phi"`. Defaults to False.

        Returns:
            None

        Side Effects:
            - Renders an interactive Plotly figure.
            - Optionally writes an image file to disk.
        """

        if len(self.solutes) == 0:
            print("Simulation has no solutes loaded")
            return

        if "phi" not in self.solutes[0].results:
            # If surface potential has not been calculated, calculate it now
            self.calculate_surface_potential()
        if solutes_names:
            if isinstance(solutes_names[0], int):
                solutes_plot = [
                    index
                    for index in range(len(self.solutes_names))
                    if index in solutes_names
                ]
            elif isinstance(solutes_names[0], str):
                solutes_plot = [
                    solute for solute in self.solutes_names if solute in solutes_names
                ]
        else:
            solutes_plot = self.solutes_names
        if len(solutes_plot) == 0:
            print("Check valid solute name/index before plotting")
            return

        print(f"Plotting solutes {solutes_plot}")

        if isosurface_potential:
            bboxes = np.array([solute.mesh.bounding_box for solute in self.solutes])
            x_min, y_min, z_min = bboxes[:, :, 0].min(axis=0)
            x_max, y_max, z_max = bboxes[:, :, 1].max(axis=0)
            extension, ref = 10, 25j
            X, Y, Z = np.mgrid[
                (x_min - extension) : (x_max + extension) : ref,
                (y_min - extension) : (y_max + extension) : ref,
                (z_min - extension) : (z_max + extension) : ref,
            ]
            coordinates = [X, Y, Z]
            potential, _ = self.calculate_potential_solvent(
                np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1), units=units
            )
            isosurface_potential_vals = (
                coordinates,
                potential,
                isosurface_potential,
            )
        else:
            isosurface_potential_vals = (None, None, None)

        plotting_tools.plot_multiple_surfaces(
            self,
            values=values,
            units=units,
            max_colorbar_scale=max_colorbar_scale,
            name=name,
            savefig=savefig,
            location=location,
            show_axis=show_axis,
            camera_view=camera_view,
            isosurface_potential_vals=isosurface_potential_vals,
            min_max_vals=min_max_vals,
            figsize=figsize,
            solutes_plot=solutes_plot,
            internal_derivative_plot=internal_derivative_plot,
        )
        return None
