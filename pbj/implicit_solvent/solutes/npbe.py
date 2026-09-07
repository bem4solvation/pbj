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
from numba import prange
from scipy.spatial import KDTree


class NPBE(Solute):
    """NPBE solute."""

    def __init__(self, solute_file_path, external_mesh_file=None, **kwargs):
        super().__init__(solute_file_path=solute_file_path, **kwargs)

        if external_mesh_file is not None:
            _, file_extension = os.path.splitext(external_mesh_file)
            if file_extension == "":  # Assume use of vert and face
                mesh_v_filepath = external_mesh_file + ".xdmf"
                mesh_s_filepath = external_mesh_file + ".off"
            else:
                filename = os.path.basename(external_mesh_file)
                filename = os.path.splitext(filename)[0]
                mesh_v_filepath = filename + ".xdmf"
                mesh_s_filepath = filename + ".off"
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
            print("No he implementado la forma de crear mallas!")
            return
            # mesh = mesh_tools.load_fem_mesh()

        with XDMFFile(MPI.COMM_WORLD, mesh_v_filepath, "r") as xdmf:
            self.mesh_v = xdmf.read_mesh(name="mesh")
        self.mesh0 = mesh_tools.import_off_mesh(mesh_s_filepath)

        self.fenics_space = dolfinx.fem.functionspace(self.mesh_v, ("CG", 1))
        self.trace_space, self.trace_matrix = fenicsx.fenics_to_bempp_trace_data(
            self.fenics_space
        )
        self.bempp_space = bempp_cl.api.function_space(self.trace_space.grid, "P", 1)
        self.bempp_space0 = bempp_cl.api.function_space(self.mesh0, "P", 1)

        coord_0 = self.mesh_v.geometry.x
        mesh_s = trimesh.load(mesh_s_filepath, process=False)
        element_diameter_ext = np.max(mesh_s.edges_unique_length)
        L_Alpha = (
            np.logical_not(
                self._contains_kdtree(mesh_s, coord_0, element_diameter_ext, k=4)
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

        solution_dirichl = self.results["phi"]
        solution_neumann = self.results["d_phi"]

        from bempp_cl.api.operators.potential.laplace import single_layer, double_layer

        slp_q = single_layer(self.bempp_space0, self.x_q.transpose())
        dlp_q = double_layer(self.bempp_space0, self.x_q.transpose())
        phi_q = slp_q * solution_neumann - dlp_q * solution_dirichl

        self.results["phir_charges"] = phi_q
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

    def _numba_classify(self, signed_distances):
        """This function simply iterates through a list of distances in parallel and labels them. Saves a little bit of memory."""
        label = np.zeros_like(signed_distances, dtype=np.uint8)
        for i in prange(label.shape[0]):
            if signed_distances[i] > 0:
                label[i] = 1
            elif signed_distances[i] < 0:
                label[i] = 0
            else:
                label[i] = 2
        return label

    def _contains_kdtree(self, mesh, points, element_diameter, k=4):
        """
        Alternative to method 'contains' of Trimesh.
        Uses a KDTree to identify the nearest neighbors and their distances.
        Points that are very close to the mesh are compared with all vertices
        of the surface mesh and checked with a signed distance approach.

        Args:
            mesh (trimesh.base.Trimesh): Surface mesh. Shape: (m, d), where 'm' is the number of vertices of the mesh and 'd' the dimension.
            points (numpy.ndarray): Voxelized mesh. Shape: (n, d), where 'm' is the number of points of the voxelized mesh and 'd' the dimension.
            element_diameter (float): Element diameter of the surface mesh.
            k (int, optional): Number of nearest neighbors to check. Defaults to 4.

        Returns:
            numpy.ndarray: Mask of shape (n, ). True if the corresponding point is inside the surface mesh.
        """
        k = 4  # Number of nearest neighbors to check
        m, d = mesh.vertices.shape  # vertices shape: m x 3
        n, _ = points.shape  # points shape: n x 3
        print(f"m: {m}, n: {n}, d: {d}, k: {k}")

        # Run KDTree to find k-nearest neighbors with their distances:
        tree = KDTree(mesh.vertices)
        distances, indices = tree.query(points, k=k)  # get nearest k points

        # Calculate dot products with the normal of each vertex of the surface
        dot_products = np.einsum(
            "knd,nkd->nk",
            points - mesh.vertices[indices.T],
            mesh.vertex_normals[indices],
        )  # Dot products with vertex normals (inside or outside the mesh?)

        # Classify points based on their dot products (all neighbors must agree):
        classification = np.zeros(n, dtype=np.int8) + 2  # Array initiated with 2
        classification[(dot_products > 0).all(axis=1)] = 0  # Outside the surface mesh
        classification[(dot_products < 0).all(axis=1)] = 1  # Inside the surface mesh

        # Identify points that could be problematic:
        tol = 1e-15
        mask_tocheck = (distances <= element_diameter + tol).any(
            axis=1
        )  # Points that are very close to the surface mesh
        mask_tocheck[classification == 2] = (
            True  # Points on which the neighbors disagreed
        )
        print("Problematic points:", mask_tocheck.sum())

        # Check problematic points: (This part can still be memory heavy with very big examples)

        if mask_tocheck.sum() > 0:
            GB_estimate, GB_limit = (
                3.5 * (mask_tocheck.sum()) * m * (10**-8),
                15,
            )  # Subdivide the list by memory size.
            subdiv = 1 + int(GB_estimate / GB_limit)
            print("Calculating signed_distances...")

            q = trimesh.proximity.ProximityQuery(mesh)
            sub_point_mask_tocheck = np.array_split(points[mask_tocheck, :], subdiv)
            signed_distances = q.signed_distance(sub_point_mask_tocheck[0])
            for j in range(len(sub_point_mask_tocheck) - 1):  # Subsection subdiv>1
                sub_j_signed_distances = q.signed_distance(
                    sub_point_mask_tocheck[j + 1]
                )
                signed_distances = np.concatenate(
                    (signed_distances, sub_j_signed_distances)
                )

            # https://trimesh.org/trimesh.proximity.html#trimesh.proximity.ProximityQuery.signed_distance
            print("Signed_distances calculated")
            # 0: outside, 1: inside, 2: signed_distance = 0
            classification[mask_tocheck] = self._numba_classify(signed_distances)

        # Verify there are no problems and return:
        # assert (classification != 2).all(), "Some classification values are still 2"
        return classification.astype(bool)
