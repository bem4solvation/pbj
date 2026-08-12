from .solute_common import Solute
import pbj.mesh.charge_tools as charge_tools
import pbj.mesh.mesh_tools as mesh_tools
import os
import time
import numpy as np
import bempp_cl as bempp
import pbj.implicit_solvent.pb_formulation.lpbe as pb_formulations


class LPBE(Solute):
    """LPBE solute specialization."""

    def __init__(
        self,
        solute_file_path,
        solute_type="lpbe",
        formulation="direct",
        external_mesh_file=None,
        force_field="amber",
        **kwargs
    ):

        super().__init__(
            solute_file_path=solute_file_path,
            solute_type=solute_type,
            formulation=formulation,
            **kwargs
        )

        if force_field == "amoeba":
            raise ValueError(
                "Use LPBE_AMOEBA solute class to create solute with %s" % force_field
            )
        else:
            self.force_field = force_field

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

            (
                self.mesh,
                self.q,
                self.x_q,
                self.r_q,
                self.atom_name,
                self.res_name,
                self.res_num,
            ) = charge_tools.generate_msms_mesh_import_charges(self)

        # Setup Dirichlet and Neumann spaces to use, save these as object vars
        dirichl_space = bempp.api.function_space(self.mesh, "P", 1)
        neumann_space = dirichl_space
        self.dirichl_space = dirichl_space
        self.neumann_space = neumann_space

    @property
    def stern_mesh_density(self):
        return self._stern_mesh_density

    @stern_mesh_density.setter
    def stern_mesh_density(self, value):
        self._stern_mesh_density = value
        self.stern_mesh_density_ratio = value / self.sas_mesh_density
        pb_formulations.direct_stern.create_stern_mesh(self)

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
            units (str, optional): Unit identifier passed to `convert_units` for the
                returned energy values. Defaults to "kcal_mol".

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

        Args:
            units (str, optional): Unit identifier passed to `convert_units` for the
                computed electrostatic solvation energy. Defaults to "kcal_mol".

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

        start_time = time.time()

        solution_dirichl = self.results["phi"]
        solution_neumann = self.results["d_phi"]

        from bempp_cl.api.operators.potential.laplace import single_layer, double_layer

        slp_q = single_layer(self.neumann_space, self.x_q.transpose())
        dlp_q = double_layer(self.dirichl_space, self.x_q.transpose())
        phi_q = slp_q * solution_neumann - dlp_q * solution_dirichl

        self.results["phir_charges"] = phi_q

        # total solvation energy applying constant to get units [kcal/mol]
        unit_conversion, unit_label = charge_tools.convert_units(
            units, magnitude="energy"
        )
        total_energy = 0.5 * unit_conversion * np.sum(self.q * phi_q).real
        self.results["electrostatic_solvation_energy"] = total_energy
        self.results["electrostatic_solvation_energy_units"] = unit_label
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
            units (str, optional): Unit identifier passed to `convert_units` for the
                computed nonpolar solvation energy. Defaults to "kcal_mol".

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
            units (str, optional): Unit identifier passed to `convert_units` for the
                computed cavity energy. Defaults to "kcal_mol".

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
        unit_conversion, unit_label = charge_tools.convert_units(
            units, magnitude="energy"
        )
        unit_conversion_base, _ = charge_tools.convert_units(
            "kcal_mol", magnitude="energy"
        )
        factor_units = unit_conversion / unit_conversion_base
        self.results["cavity_energy"] = factor_units * cavity_energy
        self.results["cavity_energy_units"] = unit_label

    def calculate_dispersion_energy(self, sas_mesh_density=None, units="kcal_mol"):
        r"""Calculate the nonpolar dispersion energy based on the Solvent Accessible Surface Area (SASA).

        Computes the hydrophobic/nonpolar dispersion contribution to the solvation free
        energy using a linear relationship with the SASA ($E_{\text{disp}} = \gamma \cdot \text{SASA} + b$).
        The SASA is determined by summing the individual element areas (`volumes`) of the SAS mesh.

        Args:
            sas_mesh_density (float, optional): Density parameter passed to `create_sas_mesh`
                                                if the SAS mesh hasn't been generated yet.
                                                Defaults to None.
            units (str, optional): Unit identifier passed to `convert_units` for the
                computed dispersion energy. Defaults to "kcal_mol".

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
        unit_conversion, unit_label = charge_tools.convert_units(
            units, magnitude="energy"
        )
        unit_conversion_base, _ = charge_tools.convert_units(
            "kcal_mol", magnitude="energy"
        )
        factor_units = unit_conversion / unit_conversion_base
        self.results["dispersion_energy"] = factor_units * dispersion_energy
        self.results["dispersion_energy_units"] = unit_label

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
            units (str, optional): Unit identifier passed to `convert_units` for the
                calculated force values. Defaults to "kcal_molA".

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

        unit_conversion, unit_label = charge_tools.convert_units(
            units, magnitude="force"
        )

        f_reac = unit_conversion * -np.transpose(np.transpose(dphidr) * self.q)
        f_reactotal = np.sum(f_reac, axis=0)

        self.results["f_qf_charges"] = f_reac
        self.results["f_qf"] = f_reactotal
        self.results["f_qf_charges_units"] = unit_label
        self.results["f_qf_units"] = unit_label
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
            units (str, optional): Unit identifier passed to `convert_units` for the
                computed boundary force values. Defaults to "kcal_molA".

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

        unit_conversion, unit_label = charge_tools.convert_units(
            units, magnitude="force"
        )
        dS = np.transpose(np.transpose(self.mesh.normals) * self.mesh.volumes)

        if fdb_approx:
            # Dielectric boundary force
            f_db = (
                -0.5
                * unit_conversion
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

                f_db += unit_conversion * F

        # Ionic boundary force
        f_ib = (
            -0.5
            * unit_conversion
            * (self.ep_ex)
            * (self.kappa**2)
            * np.sum(np.transpose(np.transpose(dS) * phi[0] ** 2), axis=0)
        )

        self.results["f_db"] = f_db
        self.results["f_ib"] = f_ib
        self.results["f_db_units"] = unit_label
        self.results["f_ib_units"] = unit_label
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
            units (str, optional): Unit identifier passed to `convert_units` for the
                calculated solvation force values. Defaults to "kcal_molA".

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
            if self.results["f_qf_units"] != units:
                self.calculate_charges_forces(units=units)

            self.calculate_boundary_forces(fdb_approx=fdb_approx, units=units)
            _, unit_label = charge_tools.convert_units(units, magnitude="force")
            start_time = time.time()

            f_solv = np.zeros([3])
            f_qf = self.results["f_qf"]
            f_db = self.results["f_db"]
            f_ib = self.results["f_ib"]

            f_solv = f_qf + f_db + f_ib
            self.results["f_solv"] = f_solv
            self.results["f_solv_units"] = unit_label
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
                self.calculate_boundary_forces(units=units)

            if self.results["f_ib_units"] != units:
                self.calculate_boundary_forces(units=units)

            start_time = time.time()

            N_elements = self.mesh.number_of_elements
            P_normal = np.zeros([N_elements])
            phi_vertex = self.results["phi"].coefficients
            ep_hat = self.ep_in / self.ep_ex
            dphi_centers = (
                ep_hat * self.results["d_phi"].evaluate_on_element_centers()[0]
            )
            total_force = np.zeros(3)
            unit_conversion, unit_label = charge_tools.convert_units(
                units, magnitude="force"
            )

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

            self.results["P_normal"] = unit_conversion * P_normal
            self.results["P_normal_units"] = unit_label
            self.results["f_solv"] = (
                unit_conversion * total_force + self.results["f_ib"]
            )
            self.results["f_solv_units"] = unit_label
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
