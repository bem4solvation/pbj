from .solute_common import Solute
import pbj.implicit_solvent.pb_formulation.lpbe_amoeba as pb_formulations
import pbj.mesh.charge_tools as charge_tools
import pbj.mesh.mesh_tools as mesh_tools
import os
import bempp_cl as bempp
import bempp_cl.api
import numpy as np
import time
from numba import jit


class LPBE_AMOEBA(Solute):
    """LPBE_AMOEBA solute specialization."""

    def __init__(
        self,
        solute_file_path,
        external_mesh_file=None,
        solute_type="lpbe_amoeba",
        force_field="amoeba",
        radius_keyword="solute",
        solute_radius_type="PB",
        formulation="direct",
        **kwargs,
    ):
        super().__init__(
            solute_file_path=solute_file_path,
            solute_type=solute_type,
            formulation=formulation,
            **kwargs,
        )

        if force_field != "amoeba":
            raise ValueError(
                "Use LPBE solute class to create solute with %s" % force_field
            )
        else:
            self.force_field = force_field

        self.formulation_object = getattr(pb_formulations, self.pb_formulation, None)
        if self.formulation_object is None:
            raise ValueError("Unrecognised formulation type %s" % self.pb_formulation)

        self.radius_keyword = radius_keyword
        self.solute_radius_type = solute_radius_type

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

        else:  # Generate mesh from given pdb or pqr, and import charges at the same time

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

        # Setup Dirichlet and Neumann spaces to use, save these as object vars
        dirichl_space = bempp.api.function_space(self.mesh, "P", 1)
        neumann_space = dirichl_space
        self.dirichl_space = dirichl_space
        self.neumann_space = neumann_space

    def calculate_solvation_energy(
        self, electrostatic_energy=True, nonpolar_energy=False, units="kcal_mol"
    ):

        if not electrostatic_energy:
            print("Non-electrostatic solvation energy calculation not implemented yet")
            return

        if nonpolar_energy:
            print("Nonpolar solvation energy calculation not implemented yet")
            return

        start_time = time.time()

        if "phi" not in self.results:
            print(
                "Please compute surface potential first with simulation.calculate_potentials()"
            )
            return

        q = self.q
        d = self.d
        Q = self.Q

        solution_dirichl = self.results["phi"]
        solution_neumann = self.results["d_phi"]

        from bempp_cl.api.operators.potential.laplace import single_layer, double_layer

        slp_q = single_layer(self.neumann_space, self.x_q.transpose())
        dlp_q = double_layer(self.dirichl_space, self.x_q.transpose())
        phi_q = slp_q * solution_neumann - dlp_q * solution_dirichl

        self.results["phir_charges"] = phi_q[0, :]

        if "gradphir_charges" not in self.results:
            self.calculate_gradient_field()

        if "gradgradphir_charges" not in self.results:
            self.calculate_gradgradient_field()

        # total solvation energy applying constant to get units [kcal/mol]
        q_aux = 0
        d_aux = 0
        Q_aux = 0

        dphi_q = self.results["gradphir_charges"]
        ddphi_q = self.results["gradgradphir_charges"]

        for i in range(len(q)):
            q_aux += q[i] * phi_q[0, i]
            for j in range(3):
                d_aux += d[i, j] * dphi_q[i, j]
                for k in range(3):
                    Q_aux += Q[i, j, k] * ddphi_q[i, j, k] / 6.0

        unit_conversion, unit_label = charge_tools.convert_units(
            units, magnitude="energy"
        )
        solvent_energy = 0.5 * unit_conversion * (q_aux + d_aux + Q_aux)
        coulomb_energy_dissolved = self.calculate_coulomb_energy_multipole(
            state="dissolved", units=units
        )

        self.calculate_induced_dipole_vacuum()
        coulomb_energy_vacuum = self.calculate_coulomb_energy_multipole(
            state="vacuum", units=units
        )

        self.results["electrostatic_solvation_energy"] = (
            solvent_energy + coulomb_energy_dissolved - coulomb_energy_vacuum
        )
        self.results["coulomb_energy_dissolved"] = coulomb_energy_dissolved
        self.results["coulomb_energy_vacuum"] = coulomb_energy_vacuum
        self.results["electrostatic_solvation_energy_units"] = unit_label
        self.timings["time_calc_energy"] = time.time() - start_time

        if self.print_times:
            print(
                "It took ",
                self.timings["time_calc_energy"],
                " seconds to compute the solvation energy",
            )

    def calculate_coulomb_energy_multipole(self, state, units="kcal/mol"):
        """
        Calculates the Coulomb energy

        state: (string) dissolved or vacuum, to choose which induced dipole to use
        """

        q = self.q
        d = self.d
        Q = self.Q

        # phi, dphi and ddphi from permanent multipoles
        phi_perm = self.calculate_coulomb_phi_multipole()
        flag_polar_group = False
        dphi_perm = self.calculate_coulomb_dphi_multipole(
            flag_polar_group
        )  # Recalculate for energy as flag = False
        ddphi_perm = self.calculate_coulomb_ddphi_multipole()

        self.results["phi_perm_multipoles"] = phi_perm
        self.results["gradphi_perm_multipoles"] = dphi_perm
        self.results["gradgradphi_perm_multipoles"] = ddphi_perm

        # phi, dphi and ddphi from induced dipoles
        phi_thole = self.calculate_coulomb_phi_multipole_Thole(state)
        dphi_thole = self.calculate_coulomb_dphi_multipole_Thole(state)
        ddphi_thole = self.calculate_coulomb_ddphi_multipole_Thole(state)

        self.results["phi_induced_dipole_" + state] = phi_thole
        self.results["gradphi_induced_dipole_" + state] = dphi_thole
        self.results["gradgradphi_induced_dipole_" + state] = ddphi_thole

        phi = phi_perm + phi_thole
        dphi = dphi_perm + dphi_thole
        ddphi = ddphi_perm + ddphi_thole

        point_energy = (
            q[:] * phi[:]
            + np.sum(d[:] * dphi[:], axis=1)
            + (np.sum(np.sum(Q[:] * ddphi[:], axis=1), axis=1)) / 6.0
        )

        unit_conversion, _ = charge_tools.convert_units(units, magnitude="energy")

        coulomb_energy = (
            sum(point_energy) * 0.5 * unit_conversion / (4 * np.pi * self.ep_in)
        )

        return coulomb_energy

    def calculate_induced_dipole_dissolved(self):

        N = len(self.x_q)

        p12scale_temp = self.p12scale
        p13scale_temp = self.p13scale

        u12scale = 1.0
        u13scale = 1.0

        self.p12scale = u12scale
        self.p13scale = u13scale  # scaling for induced dipole calculation

        alphaxx = self.alpha[:, 0, 0]

        if "d_phi_coulomb_multipole" not in self.results:
            dphi_perm = self.calculate_coulomb_dphi_multipole()
            self.results["d_phi_coulomb_multipole"] = dphi_perm

        dphi_Thole = self.calculate_coulomb_dphi_multipole_Thole(state="dissolved")

        self.p12scale = p12scale_temp
        self.p13scale = p13scale_temp

        dphi_coul = self.results["d_phi_coulomb_multipole"] + dphi_Thole
        dphi_reac = self.results["gradphir_charges"]
        d_induced = self.results["induced_dipole"]

        for i in range(N):

            E_total = (dphi_coul[i] / self.ep_in + 4 * np.pi * dphi_reac[i]) * -1
            d_induced[i] = (
                d_induced[i] * (1 - self.SOR) + np.dot(alphaxx[i], E_total) * self.SOR
            )

        self.d_induced[:] = d_induced[:]
        self.results["induced_dipole"] = d_induced

    def calculate_induced_dipole_vacuum(self):

        N = len(self.x_q)

        u12scale = 1.0
        u13scale = 1.0

        alphaxx = self.alpha[:, 0, 0]

        if "d_phi_coulomb_multipole" not in self.results:
            dphi_perm = self.calculate_coulomb_dphi_multipole()
            self.results["d_phi_coulomb_multipole"] = dphi_perm

        d_induced = np.zeros_like(self.d)
        self.results["induced_dipole_vacuum"] = d_induced

        self.results["dipole_iter_count_vacuum"] = 0

        induced_dipole_residual = 1.0
        dipole_iter_count_vacuum = 1

        bempp.api.log(
            "PBJ: Starting self-consistent interations for induced dipole in vacuum state"
        )

        while induced_dipole_residual > self.induced_dipole_iter_tol:

            p12scale_temp = self.p12scale
            p13scale_temp = self.p13scale

            self.p12scale = u12scale
            self.p13scale = u13scale  # scaling for induced dipole calculation

            dphi_Thole = self.calculate_coulomb_dphi_multipole_Thole(state="vacuum")

            self.p12scale = p12scale_temp
            self.p13scale = p13scale_temp

            dphi_coul = self.results["d_phi_coulomb_multipole"] + dphi_Thole

            d_induced_prev = d_induced.copy()

            for i in range(N):
                E_total = (dphi_coul[i] / self.ep_in) * -1
                d_induced[i] = (
                    d_induced[i] * (1 - self.SOR)
                    + np.dot(alphaxx[i], E_total) * self.SOR
                )

            induced_dipole_residual = np.max(
                np.sqrt(
                    np.sum((np.linalg.norm(d_induced_prev - d_induced, axis=1)) ** 2)
                    / len(d_induced)
                )
            )

            bempp.api.log(
                "PBJ: Vacuum induced dipole iteration %i -> residual: %s"
                % (dipole_iter_count_vacuum, induced_dipole_residual)
            )

            dipole_iter_count_vacuum += 1

        self.results["induced_dipole_vacuum"] = d_induced

    def calculate_coulomb_phi_multipole(self):
        """
        Calculate the potential due to the permanent multipoles
        """
        xq = self.x_q
        q = self.q
        d = self.d
        Q = self.Q

        phi = self._calculate_coulomb_phi_multipole(xq, q, d, Q)

        return phi

    @staticmethod
    @jit(nopython=True)
    def _calculate_coulomb_phi_multipole(xq, q, d, Q):
        """
        Performs calculation of potencial due to permanent multipoles with jit
        """
        N = len(xq)
        eps = 1e-15
        phi = np.zeros(N)

        T2 = np.zeros((N - 1, 3, 3))

        for i in range(N):

            Ri = xq[i] - xq
            Rnorm = np.sqrt(np.sum((Ri * Ri), axis=1) + eps * eps)

            Ri = np.delete(Ri, (3 * i, 3 * i + 1, 3 * i + 2)).reshape((N - 1, 3))
            Rnorm = np.delete(Rnorm, i)
            q_temp = np.delete(q, i)
            d_temp = np.delete(d, (3 * i, 3 * i + 1, 3 * i + 2)).reshape((N - 1, 3))
            Q_temp = np.delete(
                Q,
                (
                    9 * i,
                    9 * i + 1,
                    9 * i + 2,
                    9 * i + 3,
                    9 * i + 4,
                    9 * i + 5,
                    9 * i + 6,
                    9 * i + 7,
                    9 * i + 8,
                ),
            ).reshape((N - 1, 3, 3))

            T0 = 1.0 / Rnorm[:]
            T1 = np.transpose(Ri.transpose() / Rnorm**3)
            T2[:, :, :] = (
                np.ones((N - 1, 3, 3))[:]
                * Ri.reshape((N - 1, 1, 3))
                * np.transpose(
                    np.ones((N - 1, 3, 3)) * Ri.reshape((N - 1, 1, 3)), (0, 2, 1)
                )
                / Rnorm.reshape((N - 1, 1, 1)) ** 5
            )
            phi[i] = (
                np.sum(q_temp[:] * T0[:])
                + np.sum(T1[:] * d_temp[:])
                + 0.5 * np.sum(np.sum(T2[:] * Q_temp[:], axis=1))
            )

        return phi

    def calculate_coulomb_dphi_multipole(self, flag_polar_group=True):
        """
        Calculates the first derivative of the potential due to the permanent multipoles

        flag_polar_group: (bool) consider polar groups in calculation
        """

        xq = self.x_q
        q = self.q
        d = self.d
        Q = self.Q
        alphaxx = self.alpha[:, 0, 0]
        thole = self.thole
        polar_group = self.polar_group

        dphi = self._calculate_coulomb_dphi_multipole(
            xq, q, d, Q, alphaxx, thole, polar_group, flag_polar_group
        )

        return dphi

    @staticmethod
    @jit(nopython=True, parallel=False, error_model="numpy", fastmath=True)
    def _calculate_coulomb_dphi_multipole(
        xq, q, d, Q, alphaxx, thole, polar_group, flag_polar_group
    ):
        """
        Calculates the first derivative of the potential due to the permanent multipoles
        with numba jit

        flag_polar_group: (bool) consider polar groups in calculation
        """

        N = len(xq)
        T1 = np.zeros((3))
        T2 = np.zeros((3, 3))
        eps = 1e-15

        scale3 = 1.0
        scale5 = 1.0
        scale7 = 1.0

        dphi = np.zeros((N, 3))

        for i in range(N):

            aux = np.zeros((3))

            Ri = xq[i] - xq
            Rnorm = np.sqrt(np.sum((Ri * Ri), axis=1) + eps * eps)

            for j in np.where(Rnorm > 1e-12)[0]:

                R3 = Rnorm[j] ** 3
                R5 = Rnorm[j] ** 5
                R7 = Rnorm[j] ** 7

                if not flag_polar_group:

                    not_same_polar_group = True

                else:

                    gamma = min(thole[i], thole[j])
                    damp = (alphaxx[i] * alphaxx[j]) ** 0.16666667
                    damp += 1e-12
                    damp = -1 * gamma * (R3 / (damp * damp * damp))
                    expdamp = np.exp(damp)

                    scale3 = 1 - expdamp
                    scale5 = 1 - expdamp * (1 - damp)
                    scale7 = 1 - expdamp * (1 - damp + 0.6 * damp * damp)

                    if polar_group[i] != polar_group[j]:

                        not_same_polar_group = True

                    else:

                        not_same_polar_group = False

                if not_same_polar_group:

                    for k in range(3):

                        T0 = -Ri[j, k] / R3 * scale3

                        for ll in range(3):

                            dkl = (k == ll) * 1.0

                            T1[ll] = (
                                dkl / R3 * scale3
                                - 3 * Ri[j, k] * Ri[j, ll] / R5 * scale5
                            )

                            for m in range(3):

                                dkm = (k == m) * 1.0
                                T2[ll][m] = (
                                    dkm * Ri[j, ll] + dkl * Ri[j, m]
                                ) / R5 * scale5 - 5 * Ri[j, ll] * Ri[j, m] * Ri[
                                    j, k
                                ] / R7 * scale7

                        aux[k] += (
                            T0 * q[j]
                            + np.sum(T1 * d[j])
                            + 0.5
                            * np.sum(np.sum(T2[:, :] * Q[j, :, :], axis=1), axis=0)
                        )

            dphi[i, :] += aux[:]

        return dphi

    def calculate_coulomb_ddphi_multipole(self):
        """
        Calculates the second derivative of the electrostatic potential of the permantent multipoles
        """
        xq = self.x_q
        q = self.q
        d = self.d
        Q = self.Q

        ddphi = self._calculate_coulomb_ddphi_multipole(xq, q, d, Q)

        return ddphi

    @staticmethod
    @jit(nopython=True, parallel=False, error_model="numpy", fastmath=True)
    def _calculate_coulomb_ddphi_multipole(xq, q, d, Q):
        """
        Calculates the second derivative of the electrostatic potential of the permantent multipoles
        with numba jit
        """
        T1 = np.zeros((3))
        T2 = np.zeros((3, 3))

        eps = 1e-15

        N = len(xq)

        ddphi = np.zeros((N, 3, 3))

        for i in range(N):

            aux = np.zeros((3, 3))

            Ri = xq[i] - xq
            Rnorm = np.sqrt(np.sum((Ri * Ri), axis=1) + eps * eps)

            for j in np.where(Rnorm > 1e-12)[0]:

                R3 = Rnorm[j] ** 3
                R5 = Rnorm[j] ** 5
                R7 = Rnorm[j] ** 7
                R9 = R3**3

                for k in range(3):

                    for ll in range(3):

                        dkl = (k == ll) * 1.0
                        T0 = -dkl / R3 + 3 * Ri[j, k] * Ri[j, ll] / R5

                        for m in range(3):

                            dkm = (k == m) * 1.0
                            dlm = (ll == m) * 1.0

                            T1[m] = (
                                -3
                                * (dkm * Ri[j, ll] + dkl * Ri[j, m] + dlm * Ri[j, k])
                                / R5
                                + 15 * Ri[j, ll] * Ri[j, m] * Ri[j, k] / R7
                            )

                            for n in range(3):

                                dkn = (k == n) * 1.0
                                dln = (ll == n) * 1.0

                                T2[m][n] = (
                                    35 * Ri[j, k] * Ri[j, ll] * Ri[j, m] * Ri[j, n] / R9
                                    - 5
                                    * (
                                        Ri[j, m] * Ri[j, n] * dkl
                                        + Ri[j, ll] * Ri[j, n] * dkm
                                        + Ri[j, m] * Ri[j, ll] * dkn
                                        + Ri[j, k] * Ri[j, n] * dlm
                                        + Ri[j, m] * Ri[j, k] * dln
                                    )
                                    / R7
                                    + (dkm * dln + dlm * dkn) / R5
                                )

                        aux[k][ll] += (
                            T0 * q[j]
                            + np.sum(T1[:] * d[j, :])
                            + 0.5
                            * np.sum(np.sum(T2[:, :] * Q[j, :, :], axis=1), axis=0)
                        )

            ddphi[i, :, :] += aux[:, :]

        return ddphi

    def calculate_coulomb_phi_multipole_Thole(self, state):
        """
        Calculates the potential due to the induced dipoles according to Thole

        state: (string) dissolved or vacuum, to choose which induced dipole to use
        """

        xq = self.x_q
        if state == "dissolved":
            induced_dipole = self.results["induced_dipole"]
        else:
            induced_dipole = self.results["induced_dipole_vacuum"]

        thole = self.thole
        alphaxx = self.alpha[:, 0, 0]
        connections_12 = self.connections_12
        pointer_connections_12 = self.pointer_connections_12
        connections_13 = self.connections_13
        pointer_connections_13 = self.pointer_connections_13
        p12scale = self.p12scale
        p13scale = self.p13scale

        phi = self._calculate_coulomb_phi_multipole_Thole(
            xq,
            induced_dipole,
            thole,
            alphaxx,
            connections_12,
            pointer_connections_12,
            connections_13,
            pointer_connections_13,
            p12scale,
            p13scale,
        )

        return phi

    @staticmethod
    @jit(nopython=True, parallel=False, error_model="numpy", fastmath=True)
    def _calculate_coulomb_phi_multipole_Thole(
        xq,
        induced_dipole,
        thole,
        alphaxx,
        connections_12,
        pointer_connections_12,
        connections_13,
        pointer_connections_13,
        p12scale,
        p13scale,
    ):
        """
        Calculates the potential due to the induced dipoles according to Thole
        with numba jit

        """

        eps = 1e-15
        T1 = np.zeros((3))

        N = len(xq)

        phi = np.zeros((N))

        for i in range(N):

            aux = 0.0
            start_12 = pointer_connections_12[i]
            stop_12 = pointer_connections_12[i + 1]
            start_13 = pointer_connections_13[i]
            stop_13 = pointer_connections_13[i + 1]

            Ri = xq[i] - xq

            r = 1.0 / np.sqrt(np.sum((Ri * Ri), axis=1) + eps * eps)

            for j in np.where(r < 1e12)[0]:

                pscale = 1.0

                for ii in range(start_12, stop_12):

                    if connections_12[ii] == j:

                        pscale = p12scale

                for ii in range(start_13, stop_13):

                    if connections_13[ii] == j:

                        pscale = p13scale

                r3 = r[j] ** 3

                gamma = min(thole[i], thole[j])
                damp = (alphaxx[i] * alphaxx[j]) ** 0.16666667
                damp += 1e-12
                damp = -gamma * (1 / (r3 * damp**3))
                expdamp = np.exp(damp)

                scale3 = 1 - expdamp

                for k in range(3):

                    T1[k] = Ri[j, k] * r3 * scale3 * pscale

                aux += np.sum(T1[:] * induced_dipole[j, :])

            phi[i] += aux

        return phi

    def calculate_coulomb_dphi_multipole_Thole(self, state):
        """
        Calculates the derivative of the potential due to the induced dipoles according to Thole

        state: (string) dissolved or vacuum, to choose which induced dipole to use
        """

        xq = self.x_q
        if state == "dissolved":
            induced_dipole = self.results["induced_dipole"]
        elif state == "vacuum":
            induced_dipole = self.results["induced_dipole_vacuum"]
        else:
            print("Cannot understand state")

        thole = self.thole
        connections_12 = self.connections_12
        pointer_connections_12 = self.pointer_connections_12
        connections_13 = self.connections_13
        pointer_connections_13 = self.pointer_connections_13
        p12scale = self.p12scale
        p13scale = self.p13scale
        alphaxx = self.alpha[:, 0, 0]

        dphi = self._calculate_coulomb_dphi_multipole_Thole(
            xq,
            induced_dipole,
            thole,
            alphaxx,
            connections_12,
            pointer_connections_12,
            connections_13,
            pointer_connections_13,
            p12scale,
            p13scale,
        )

        return dphi

    @staticmethod
    @jit(nopython=True, parallel=False, error_model="numpy", fastmath=True)
    def _calculate_coulomb_dphi_multipole_Thole(
        xq,
        induced_dipole,
        thole,
        alphaxx,
        connections_12,
        pointer_connections_12,
        connections_13,
        pointer_connections_13,
        p12scale,
        p13scale,
    ):
        """
        Calculates the derivative of the potential due to the induced dipoles according to Thole
        with numba jit
        """

        eps = 1e-15
        T1 = np.zeros((3))

        N = len(xq)

        dphi = np.zeros((N, 3))

        for i in range(N):

            aux = np.zeros((3))

            start_12 = pointer_connections_12[i]
            stop_12 = pointer_connections_12[i + 1]
            start_13 = pointer_connections_13[i]
            stop_13 = pointer_connections_13[i + 1]

            Ri = xq[i] - xq
            r = 1.0 / np.sqrt(np.sum((Ri * Ri), axis=1) + eps * eps)

            for j in np.where(r < 1e12)[0]:

                pscale = 1.0

                for ii in range(start_12, stop_12):

                    if connections_12[ii] == j:

                        pscale = p12scale

                for ii in range(start_13, stop_13):

                    if connections_13[ii] == j:

                        pscale = p13scale

                r3 = r[j] ** 3
                r5 = r[j] ** 5

                gamma = min(thole[i], thole[j])
                damp = (alphaxx[i] * alphaxx[j]) ** 0.16666667
                damp += 1e-12
                damp = -gamma * (1 / (r3 * damp**3))
                expdamp = np.exp(damp)

                scale3 = 1 - expdamp
                scale5 = 1 - expdamp * (1 - damp)

                for k in range(3):

                    for ll in range(3):

                        dkl = (k == ll) * 1.0
                        T1[ll] = (
                            scale3 * dkl * r3 * pscale
                            - scale5 * 3 * Ri[j, k] * Ri[j, ll] * r5 * pscale
                        )

                    aux[k] += np.sum(T1[:] * induced_dipole[j, :])

            dphi[i, :] += aux[:]

        return dphi

    def calculate_coulomb_ddphi_multipole_Thole(self, state):
        """
        Calculates the second derivative of the potential due to the induced dipoles according to Thole

        state: (string) dissolved or vacuum, to choose which induced dipole to use
        """

        xq = self.x_q

        if state == "dissolved":
            induced_dipole = self.results["induced_dipole"]
        else:
            induced_dipole = self.results["induced_dipole_vacuum"]

        thole = self.thole
        connections_12 = self.connections_12
        pointer_connections_12 = self.pointer_connections_12
        connections_13 = self.connections_13
        pointer_connections_13 = self.pointer_connections_13
        p12scale = self.p12scale
        p13scale = self.p13scale
        alphaxx = self.alpha[:, 0, 0]

        ddphi = self._calculate_coulomb_ddphi_multipole_Thole(
            xq,
            induced_dipole,
            thole,
            alphaxx,
            connections_12,
            pointer_connections_12,
            connections_13,
            pointer_connections_13,
            p12scale,
            p13scale,
        )

        return ddphi

    @staticmethod
    @jit(nopython=True, parallel=False, error_model="numpy", fastmath=True)
    def _calculate_coulomb_ddphi_multipole_Thole(
        xq,
        induced_dipole,
        thole,
        alphaxx,
        connections_12,
        pointer_connections_12,
        connections_13,
        pointer_connections_13,
        p12scale,
        p13scale,
    ):
        """
        Calculates the second derivative of the potential due to the induced dipoles according to Thole
        with numba jit
        """

        eps = 1e-15
        T1 = np.zeros((3))

        N = len(xq)

        ddphi = np.zeros((N, 3, 3))

        for i in range(N):

            aux = np.zeros((3, 3))

            start_12 = pointer_connections_12[i]
            stop_12 = pointer_connections_12[i + 1]
            start_13 = pointer_connections_13[i]
            stop_13 = pointer_connections_13[i + 1]

            Ri = xq[i] - xq
            r = 1.0 / np.sqrt(np.sum((Ri * Ri), axis=1) + eps * eps)

            for j in np.where(r < 1e12)[0]:

                pscale = 1.0

                for ii in range(start_12, stop_12):

                    if connections_12[ii] == j:

                        pscale = p12scale

                for ii in range(start_13, stop_13):

                    if connections_13[ii] == j:

                        pscale = p13scale

                r3 = r[j] ** 3
                r5 = r[j] ** 5
                r7 = r[j] ** 7

                gamma = min(thole[i], thole[j])
                damp = (alphaxx[i] * alphaxx[j]) ** 0.16666667
                damp += 1e-12
                damp = -gamma * (1 / (r3 * damp**3))
                expdamp = np.exp(damp)

                scale5 = 1 - expdamp * (1 - damp)
                scale7 = 1 - expdamp * (1 - damp + 0.6 * damp**2)

                for k in range(3):

                    for ll in range(3):

                        dkl = (k == ll) * 1.0

                        for m in range(3):

                            dkm = (k == m) * 1.0
                            dlm = (ll == m) * 1.0

                            T1[m] = (
                                -3
                                * (dkm * Ri[j, ll] + dkl * Ri[j, m] + dlm * Ri[j, k])
                                * r5
                                * scale5
                                * pscale
                                + 15
                                * Ri[j, ll]
                                * Ri[j, m]
                                * Ri[j, k]
                                * r7
                                * scale7
                                * pscale
                            )

                        aux[k][ll] += np.sum(T1[:] * induced_dipole[j, :])

            ddphi[i, :, :] += aux[:, :]

        return ddphi
