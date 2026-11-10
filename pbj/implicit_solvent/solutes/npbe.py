from .solute_common import Solute
import pbj.mesh.charge_tools as charge_tools
import pbj.mesh.mesh_tools as mesh_tools
import dolfinx
import time
import numpy as np
import bempp_cl.api
import trimesh
import os
from bempp_cl.api.external import fenicsx
from mpi4py import MPI
from dolfinx.io import XDMFFile
import ufl


class NPBE(Solute):
    """NPBE solute."""

    def __init__(
        self,
        solute_file_path,
        external_mesh_volume_file=None,
        external_mesh_file=None,
        max_volume_size=0.1,
        nonlinear_thickness=3.0,
        **kwargs,
    ):
        """Initializes an NPBE solute with its surface (BEM) and volumetric (FEM) meshes.

        When both external meshes are provided, they are imported and the charges are read
        from the structure file. Otherwise the meshes are generated with TetGen and the charges
        are imported at the same time. The constructor then builds the FEniCSx function space,
        the trace space linking FEM and BEM, the bempp boundary spaces, and `Alpha`, a FEM
        indicator function that is 1 outside the solute and 0 inside it.

        Args:
            solute_file_path (str): Path to the molecular structure file (.pdb, .pqr, or .xyz).
            external_mesh_volume_file (str, optional): Path to a precomputed volumetric mesh,
                in .xdmf or .msh (gmsh) format. Only used together with `external_mesh_file`.
                Defaults to None.
            external_mesh_file (str, optional): Path to a precomputed surface mesh in .off
                format. Only used together with `external_mesh_volume_file`. Defaults to None.
            max_volume_size (float, optional): Maximum tetrahedron volume used by TetGen when
                generating the volumetric mesh. Defaults to 0.1.
            nonlinear_thickness (float, optional): Thickness, in Angstrom, of the region around
                the solute where the nonlinear equation is solved. Defaults to 3.0.
            **kwargs: Additional keyword arguments forwarded to `Solute.__init__`
                (e.g., `mesh_density`, `solvent_radius`, `mesh_generator`, `force_field`).

        Note:
            If the volumetric mesh format is not recognized, a message is printed and the
            constructor returns early, leaving the object only partially initialized.
        """

        super().__init__(solute_file_path=solute_file_path, **kwargs)

        if external_mesh_file is not None and external_mesh_volume_file is not None:

            self.external_mesh_volume_path = external_mesh_volume_file
            self.external_mesh_path = external_mesh_file
            file_extension = os.path.splitext(self.external_mesh_volume_path)[1]
            if "xdmf" in file_extension:  # Assume use of vert and face
                with XDMFFile(
                    MPI.COMM_WORLD, self.external_mesh_volume_path, "r"
                ) as xdmf:
                    self.mesh_v = xdmf.read_mesh(name="mesh")
            elif "gmsh" in file_extension:
                mesh_data = dolfinx.io.gmsh.read_from_msh(
                    self.external_mesh_volume_path,
                    MPI.COMM_WORLD,
                    rank=0,
                    gdim=3,
                )
                self.mesh_v = mesh_data.mesh
            else:
                print(f"Unrecognizable format for volumetric mesh: {file_extension}. \
                      Only available formats are .xdmf and .msh (gmsh).")
                return

            self.mesh = mesh_tools.import_off_mesh(self.external_mesh_path)
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

        else:
            self.max_volume_size = max_volume_size
            self.nonlinear_thickness = nonlinear_thickness
            (
                self.mesh_v,
                self.mesh,
                self.mesh_out,
                self.q,
                self.x_q,
                self.r_q,
                self.atom_name,
                self.res_name,
                self.res_num,
            ) = charge_tools.generate_tetgen_mesh_import_charges(
                self
            )  # Import charges from given file

        self.fenics_space = dolfinx.fem.functionspace(self.mesh_v, ("CG", 1))
        self.trace_space, self.trace_matrix = fenicsx.fenics_to_bempp_trace_data(
            self.fenics_space
        )
        self.bempp_space = bempp_cl.api.function_space(self.trace_space.grid, "P", 1)
        self.bempp_space0 = bempp_cl.api.function_space(self.mesh, "P", 1)
        self.dirichl_space = self.bempp_space0
        self.neumann_space = self.bempp_space0

        verts = np.transpose(self.mesh.vertices)
        faces = np.transpose(self.mesh.elements)
        mesh_tri = trimesh.Trimesh(vertices=verts, faces=faces)
        coord_0 = self.mesh_v.geometry.x
        element_diameter_ext = np.max(mesh_tri.edges_unique_length)
        L_Alpha = (
            np.logical_not(
                mesh_tools.contains_kdtree(mesh_tri, coord_0, element_diameter_ext, k=4)
            )
        ).astype(int)

        self.Alpha = dolfinx.fem.Function(self.fenics_space)
        self.Alpha.x.array[:] = np.asarray(L_Alpha, dtype=np.float64)

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

        from bempp_cl.api.operators.potential.laplace import single_layer, double_layer

        slp_q = single_layer(self.bempp_space0, self.x_q.transpose())
        dlp_q = double_layer(self.bempp_space0, self.x_q.transpose())
        phi_q = slp_q * self.results["d_phi"] - dlp_q * self.results["phi"]
        self.results["phir_charges"] = phi_q

        unit_conversion, unit_label = charge_tools.convert_units(
            units, magnitude="energy"
        )
        unit_potential, _ = charge_tools.convert_units("kt")
        constant = unit_conversion / (unit_potential**2)

        u_T = unit_potential * self.results["phi_fem"]
        KI = self.ep_ex * (self.kappa**2) * self.Alpha
        NL_E = dolfinx.fem.Function(self.fenics_space)
        expr_ufl_E = ufl.sinh(u_T) * u_T + 2 - 2 * ufl.cosh(u_T)
        NL_E.interpolate(
            dolfinx.fem.Expression(
                expr_ufl_E, self.fenics_space.element.interpolation_points
            )
        )
        thickness_energy = (0.5 * constant) * dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(KI * NL_E * ufl.Measure("dx", self.mesh_v))
        )
        if np.abs(thickness_energy) == np.inf:
            # Taylor expansion of the energy terms
            expr_ufl_S = (
                u_T
                + np.power(u_T, 3) / 6
                + np.power(u_T, 5) / 120
                + np.power(u_T, 7) / 5040
                + np.power(u_T, 9) / 362880
                + np.power(u_T, 11) / 39916800
            )
            expr_ufl_C = (
                1
                + np.power(u_T, 2) / 2
                + np.power(u_T, 4) / 24
                + np.power(u_T, 6) / 720
                + np.power(u_T, 8) / 40320
                + np.power(u_T, 10) / 3628800
            )
            expr_ufl_E = expr_ufl_S * u_T + 2 - 2 * expr_ufl_C
            NL_E.interpolate(
                dolfinx.fem.Expression(
                    expr_ufl_E, self.fenics_space.element.interpolation_points
                )
            )
            thickness_energy = (0.5 * constant) * dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(KI * NL_E * ufl.Measure("dx", self.mesh_v))
            )

        charges_energy = 0.5 * unit_conversion * np.sum(self.q * phi_q).real
        self.results["electrostatic_solvation_energy_charges"] = charges_energy
        self.results["electrostatic_solvation_energy_charges_units"] = unit_label
        self.results["electrostatic_solvation_energy_thickness"] = thickness_energy
        self.results["electrostatic_solvation_energy_thickness_units"] = unit_label
        self.results["electrostatic_solvation_energy"] = (
            charges_energy + thickness_energy
        )
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
