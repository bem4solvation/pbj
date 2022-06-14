import numpy as np
import bempp.api
import os
import shutil
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
import pbj
from .common import calculate_potential_stern


def verify_parameters(self):
    return True


# Creation of stern layer mesh:
def create_stern_mesh(self):

    stern_pqr_dir = os.path.abspath("pqr_temp/")
    if self.save_mesh_build_files:
        stern_pqr_dir = self.mesh_build_files_dir

    if not os.path.exists(stern_pqr_dir):
        try:
            os.mkdir(stern_pqr_dir)
        except OSError:
            print("Creation of the directory %s failed" % stern_pqr_dir)

    stern_pqr_file = os.path.join(stern_pqr_dir, "stern_pqr.pqr")
    with open(stern_pqr_file, "w") as f:
        f.write(
            "# This is a dummy pqr file generated solely for the creation of the Stern layer, atoms' position and charge are similar to the molecule's original pqr. The rest of the info shouldn't be used as reference for the molecule's composition.\n"
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
                + str(self.r_q[index] + self.pb_formulation_stern_width)
                + "\n"
            )
    """
    if self.external_mesh_file is not None:
        raise RuntimeError("Solute was created using an external mesh file. For the stern layer mesh generation a pqr or pdb file must be used to create the solute mesh.")
    else:
    """
    if hasattr(self, "mesh_density"):
        stern_solute_object = pbj.Solute(
            stern_pqr_file,
            external_mesh_file=None,
            save_mesh_build_files=self.save_mesh_build_files,
            mesh_build_files_dir=self.mesh_build_files_dir,
            mesh_density=getattr(self, "stern_mesh_density", self.mesh_density),
            nanoshaper_grid_scale=getattr(self, "nanoshaper_grid_scale", None),
            mesh_probe_radius=self.mesh_probe_radius,
            mesh_generator=self.mesh_generator,
            print_times=self.print_times,
            force_field=self.force_field,
            formulation="direct",
        )

    else:
        stern_solute_object = pbj.Solute(
            stern_pqr_file,
            external_mesh_file=None,
            save_mesh_build_files=self.save_mesh_build_files,
            mesh_build_files_dir=self.mesh_build_files_dir,
            nanoshaper_grid_scale=getattr(self, "nanoshaper_grid_scale", None),
            mesh_probe_radius=self.mesh_probe_radius,
            mesh_generator=self.mesh_generator,
            print_times=self.print_times,
            force_field=self.force_field,
            formulation="direct",
        )

    if not self.save_mesh_build_files:
        shutil.rmtree(stern_pqr_dir)

    self.stern_object = stern_solute_object
    # return stern_solute_object


def lhs(self):
    dirichl_space_diel = self.dirichl_space
    neumann_space_diel = self.neumann_space
    e_hat_diel = self.e_hat_diel
    e_hat_stern = self.e_hat_stern
    kappa = self.kappa
    operator_assembler = self.operator_assembler

    dirichl_space_stern = self.stern_object.dirichl_space
    neumann_space_stern = self.stern_object.neumann_space

    identity_diel = sparse.identity(
        dirichl_space_diel, dirichl_space_diel, dirichl_space_diel
    )
    slp_in_diel = laplace.single_layer(
        neumann_space_diel,
        dirichl_space_diel,
        dirichl_space_diel,
        assembler=operator_assembler,
    )
    dlp_in_diel = laplace.double_layer(
        dirichl_space_diel,
        dirichl_space_diel,
        dirichl_space_diel,
        assembler=operator_assembler,
    )
    slp_out_diel = laplace.single_layer(
        dirichl_space_diel,
        dirichl_space_diel,
        dirichl_space_diel,
        assembler=operator_assembler,
    )
    dlp_out_diel = laplace.double_layer(
        dirichl_space_diel,
        dirichl_space_diel,
        dirichl_space_diel,
        assembler=operator_assembler,
    )
    slp_stern_diel = laplace.single_layer(
        neumann_space_stern,
        dirichl_space_diel,
        dirichl_space_diel,
        assembler=operator_assembler,
    )  # stern to diel
    dlp_stern_diel = laplace.double_layer(
        dirichl_space_stern,
        dirichl_space_diel,
        dirichl_space_diel,
        assembler=operator_assembler,
    )  # stern to diel
    slp_diel_stern = laplace.single_layer(
        neumann_space_diel,
        dirichl_space_stern,
        dirichl_space_stern,
        assembler=operator_assembler,
    )  # diel to stern
    dlp_diel_stern = laplace.double_layer(
        dirichl_space_diel,
        dirichl_space_stern,
        dirichl_space_stern,
        assembler=operator_assembler,
    )  # diel to stern
    identity_stern = sparse.identity(
        dirichl_space_stern, dirichl_space_stern, dirichl_space_stern
    )
    slp_in_stern = laplace.single_layer(
        neumann_space_stern,
        dirichl_space_stern,
        dirichl_space_stern,
        assembler=operator_assembler,
    )
    dlp_in_stern = laplace.double_layer(
        dirichl_space_stern,
        dirichl_space_stern,
        dirichl_space_stern,
        assembler=operator_assembler,
    )
    slp_out_stern = modified_helmholtz.single_layer(
        neumann_space_stern,
        dirichl_space_stern,
        dirichl_space_stern,
        kappa,
        assembler=operator_assembler,
    )
    dlp_out_stern = modified_helmholtz.double_layer(
        dirichl_space_stern,
        dirichl_space_stern,
        dirichl_space_stern,
        kappa,
        assembler=operator_assembler,
    )

    A = bempp.api.BlockedOperator(4, 4)

    A[0, 0] = 0.5 * identity_diel + dlp_in_diel
    A[0, 1] = -slp_in_diel
    A[1, 0] = 0.5 * identity_diel - dlp_out_diel
    A[1, 1] = slp_out_diel * e_hat_diel
    A[1, 2] = dlp_stern_diel
    A[1, 3] = -slp_stern_diel
    A[2, 0] = -dlp_diel_stern
    A[2, 1] = slp_diel_stern * e_hat_diel
    A[2, 2] = 0.5 * identity_stern + dlp_in_stern
    A[2, 3] = -slp_in_stern
    A[3, 2] = 0.5 * identity_stern - dlp_out_stern
    A[3, 3] = slp_out_stern * e_hat_stern

    self.matrices["A"] = A


def rhs(self):
    dirichl_space_diel = self.dirichl_space
    neumann_space_diel = self.neumann_space
    q = self.q
    x_q = self.x_q
    ep_in = self.ep_in
    rhs_constructor = self.rhs_constructor

    dirichl_space_stern = self.stern_object.dirichl_space
    neumann_space_stern = self.stern_object.neumann_space

    if rhs_constructor == "fmm":

        @bempp.api.callable(vectorized=True)
        def fmm_green_func(x, n, domain_index, result):
            import exafmm.laplace as _laplace

            sources = _laplace.init_sources(x_q, q)
            targets = _laplace.init_targets(x.T)
            fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename=".rhs.tmp")
            tree = _laplace.setup(sources, targets, fmm)
            values = _laplace.evaluate(tree, fmm)
            os.remove(".rhs.tmp")
            result[:] = values[:, 0] / ep_in

        # @bempp.api.real_callable
        # def zero(x, n, domain_index, result):
        #     result[0] = 0

        coefs_neumann_diel = np.zeros(neumann_space_diel.global_dof_count)
        coefs_dirichl_stern = np.zeros(dirichl_space_stern.global_dof_count)
        coefs_neumann_stern = np.zeros(neumann_space_stern.global_dof_count)

        rhs_1 = bempp.api.GridFunction(dirichl_space_diel, fun=fmm_green_func)
        rhs_2 = bempp.api.GridFunction(
            neumann_space_diel, coefficients=coefs_neumann_diel
        )
        rhs_3 = bempp.api.GridFunction(
            dirichl_space_stern, coefficients=coefs_dirichl_stern
        )
        rhs_4 = bempp.api.GridFunction(
            neumann_space_stern, coefficients=coefs_neumann_stern
        )

    else:

        @bempp.api.real_callable
        def charges_fun(x, n, domain_index, result):
            nrm = np.sqrt(
                (x[0] - x_q[:, 0]) ** 2
                + (x[1] - x_q[:, 1]) ** 2
                + (x[2] - x_q[:, 2]) ** 2
            )
            aux = np.sum(q / nrm)
            result[0] = aux / (4 * np.pi * ep_in)

        @bempp.api.real_callable
        def zero(x, n, domain_index, result):
            result[0] = 0

        rhs_1 = bempp.api.GridFunction(dirichl_space_diel, fun=charges_fun)
        rhs_2 = bempp.api.GridFunction(neumann_space_diel, fun=zero)
        rhs_3 = bempp.api.GridFunction(dirichl_space_stern, fun=zero)
        rhs_4 = bempp.api.GridFunction(neumann_space_stern, fun=zero)

    self.rhs["rhs_1"], self.rhs["rhs_2"], self.rhs["rhs_3"], self.rhs["rhs_4"] = (
        rhs_1,
        rhs_2,
        rhs_3,
        rhs_4,
    )


def block_diagonal_preconditioner(solute):
    from scipy.sparse import diags, bmat, block_diag
    from scipy.sparse.linalg import aslinearoperator
    from pbj.electrostatics.solute import matrix_to_discrete_form, rhs_to_discrete_form

    dirichl_space_diel = solute.dirichl_space
    neumann_space_diel = solute.neumann_space
    dirichl_space_stern = solute.stern_object.dirichl_space
    neumann_space_stern = solute.stern_object.neumann_space
    e_hat_diel = solute.e_hat_diel
    e_hat_stern = solute.e_hat_stern
    kappa = solute.kappa

    if not (isinstance(e_hat_diel, float) or isinstance(e_hat_diel, int)):
        e_hat_diel = e_hat_diel.weak_form().to_sparse().diagonal()

    if not (isinstance(e_hat_stern, float) or isinstance(e_hat_stern, int)):
        e_hat_stern = e_hat_stern.weak_form().to_sparse().diagonal()

    identity_diel = sparse.identity(
        dirichl_space_diel, dirichl_space_diel, dirichl_space_diel
    )
    identity_diel_diag = identity_diel.weak_form().to_sparse().diagonal()
    slp_in_diel_diag = (
        laplace.single_layer(
            neumann_space_diel,
            dirichl_space_diel,
            dirichl_space_diel,
            assembler="only_diagonal_part",
        )
        .weak_form()
        .get_diagonal()
    )
    dlp_in_diel_diag = (
        laplace.double_layer(
            dirichl_space_diel,
            dirichl_space_diel,
            dirichl_space_diel,
            assembler="only_diagonal_part",
        )
        .weak_form()
        .get_diagonal()
    )
    slp_out_diel_diag = (
        laplace.single_layer(
            dirichl_space_diel,
            dirichl_space_diel,
            dirichl_space_diel,
            assembler="only_diagonal_part",
        )
        .weak_form()
        .get_diagonal()
    )
    dlp_out_diel_diag = (
        laplace.double_layer(
            dirichl_space_diel,
            dirichl_space_diel,
            dirichl_space_diel,
            assembler="only_diagonal_part",
        )
        .weak_form()
        .get_diagonal()
    )

    identity_stern = sparse.identity(
        dirichl_space_stern, dirichl_space_stern, dirichl_space_stern
    )
    identity_stern_diag = identity_stern.weak_form().to_sparse().diagonal()
    slp_in_stern_diag = (
        laplace.single_layer(
            neumann_space_stern,
            dirichl_space_stern,
            dirichl_space_stern,
            assembler="only_diagonal_part",
        )
        .weak_form()
        .get_diagonal()
    )
    dlp_in_stern_diag = (
        laplace.double_layer(
            dirichl_space_stern,
            dirichl_space_stern,
            dirichl_space_stern,
            assembler="only_diagonal_part",
        )
        .weak_form()
        .get_diagonal()
    )
    slp_out_stern_diag = (
        modified_helmholtz.single_layer(
            neumann_space_stern,
            dirichl_space_stern,
            dirichl_space_stern,
            kappa,
            assembler="only_diagonal_part",
        )
        .weak_form()
        .get_diagonal()
    )
    dlp_out_stern_diag = (
        modified_helmholtz.double_layer(
            dirichl_space_stern,
            dirichl_space_stern,
            dirichl_space_stern,
            kappa,
            assembler="only_diagonal_part",
        )
        .weak_form()
        .get_diagonal()
    )

    diag11 = 0.5 * identity_diel_diag + dlp_in_diel_diag
    diag12 = -slp_in_diel_diag
    diag21 = 0.5 * identity_diel_diag - dlp_out_diel_diag
    diag22 = slp_out_diel_diag * e_hat_diel

    d_aux = 1 / (diag22 - diag21 * diag12 / diag11)
    diag11_inv = 1 / diag11 + 1 / diag11 * diag12 * d_aux * diag21 / diag11
    diag12_inv = -1 / diag11 * diag12 * d_aux
    diag21_inv = -d_aux * diag21 / diag11
    diag22_inv = d_aux

    diag33 = 0.5 * identity_stern_diag + dlp_in_stern_diag
    diag34 = -slp_in_stern_diag
    diag43 = 0.5 * identity_stern_diag - dlp_out_stern_diag
    diag44 = slp_out_stern_diag * e_hat_stern

    d_aux = 1 / (diag44 - diag43 * diag34 / diag33)
    diag33_inv = 1 / diag33 + 1 / diag33 * diag34 * d_aux * diag43 / diag33
    diag34_inv = -1 / diag33 * diag34 * d_aux
    diag43_inv = -d_aux * diag43 / diag33
    diag44_inv = d_aux

    A = bmat(
        [[diags(diag11_inv), diags(diag12_inv)], [diags(diag21_inv), diags(diag22_inv)]]
    )
    B = bmat(
        [[diags(diag33_inv), diags(diag34_inv)], [diags(diag43_inv), diags(diag44_inv)]]
    )

    block_mat_precond = block_diag((A, B), format="csr")

    solute.matrices["preconditioning_matrix_gmres"] = aslinearoperator(
        block_mat_precond
    )
    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [
        solute.rhs["rhs_1"],
        solute.rhs["rhs_2"],
        solute.rhs["rhs_3"],
        solute.rhs["rhs_4"],
    ]

    solute.matrices["A_discrete"] = matrix_to_discrete_form(
        solute.matrices["A_final"], "weak"
    )
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(
        solute.rhs["rhs_final"], "weak", solute.matrices["A"]
    )


def mass_matrix_preconditioner(solute):
    from pbj.electrostatics.solute import matrix_to_discrete_form, rhs_to_discrete_form

    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [
        solute.rhs["rhs_1"],
        solute.rhs["rhs_2"],
        solute.rhs["rhs_3"],
        solute.rhs["rhs_4"],
    ]
    solute.matrices["A_discrete"] = matrix_to_discrete_form(
        solute.matrices["A_final"], "strong"
    )
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(
        solute.rhs["rhs_final"], "strong", solute.matrices["A"]
    )


def calculate_potential(self, rerun_all):
    ep_stern = getattr(self, "ep_stern", self.ep_ex)
    self.ep_stern = ep_stern
    self.e_hat_diel = self.ep_in / self.ep_stern
    self.e_hat_stern = self.ep_stern / self.ep_ex
    if self.stern_object is None:
        create_stern_mesh(self)
    calculate_potential_stern(self, rerun_all)
