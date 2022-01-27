import numpy as np
import bempp.api
import os
import shutil
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
import pbj

def verify_parameters(self):
    return True

#Creation of stern layer mesh:
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
        f.write("# This is a dummy pqr file generated solely for the creation of the Stern layer, atoms' position and charge are similar to the molecule's original pqr. The rest of the info shouldn't be used as reference for the molecule's composition.\n")
        for index in range(len(self.r_q)):
            f.write("ATOM      #  #   ###     #      "+str(self.x_q[index][0]) + " " + str(self.x_q[index][1]) + " " + str(self.x_q[index][2]) + " " + str(self.q[index]) + " " + str(self.r_q[index]+self.pb_formulation_stern_width) + "\n") 
    """
    if self.external_mesh_file is not None:
        raise RuntimeError("Solute was created using an external mesh file. For the stern layer mesh generation a pqr or pdb file must be used to create the solute mesh.")
    else:
    """    
    stern_solute_object = pbj.Solute(stern_pqr_file,
        external_mesh_file = None,
        save_mesh_build_files = self.save_mesh_build_files,
        mesh_build_files_dir = self.mesh_build_files_dir,
        mesh_density = self.mesh_density,
        nanoshaper_grid_scale = getattr(self, 'nanoshaper_grid_scale', None),
        mesh_probe_radius = self.mesh_probe_radius,
        mesh_generator = self.mesh_generator,
        print_times = self.print_times,
        force_field = self.force_field,
        formulation = 'direct')

    if not self.save_mesh_build_files:
        shutil.rmtree(stern_pqr_dir)

    self.stern_object = stern_solute_object
    #return stern_solute_object

def lhs(self):
    dirichl_space_diel = self.dirichl_space
    neumann_space_diel = self.neumann_space
    ep_in = self.ep_in
    ep_stern = getattr(self, ep_stern, self.ep_out)
    self.ep_stern = ep_stern
    ep_out = self.ep_ex
    kappa = self.kappa
    operator_assembler = self.operator_assembler

    create_stern_mesh(self)
    dirichl_space_stern = self.stern_object.dirichl_space
    neumann_space_stern = self.stern_object.neumann_space

    
    identity_diel = sparse.identity(dirichl_space_diel, dirichl_space_diel, dirichl_space_diel)
    slp_in_diel = laplace.single_layer(
        neumann_space_diel, dirichl_space_diel, dirichl_space_diel, assembler=operator_assembler
    )
    dlp_in_diel = laplace.double_layer(
        dirichl_space_diel, dirichl_space_diel, dirichl_space_diel, assembler=operator_assembler
    )
    slp_out_diel = laplace.single_layer(
        dirichl_space_diel, dirichl_space_diel, dirichl_space_diel, assembler=operator_assembler
    )
    dlp_out_diel = laplace.double_layer(
        dirichl_space_diel, dirichl_space_diel, dirichl_space_diel, assembler=operator_assembler
    )
    slp_stern_diel = laplace.single_layer(
        neumann_space_stern, dirichl_space_diel, dirichl_space_diel, assembler=operator_assembler
    ) #stern to diel
    dlp_stern_diel = laplace.double_layer(
        dirichl_space_stern, dirichl_space_diel, dirichl_space_diel, assembler=operator_assembler
    ) #stern to diel
    slp_diel_stern = laplace.single_layer(
        neumann_space_diel, dirichl_space_stern, dirichl_space_stern, assembler=operator_assembler
    ) #diel to stern
    dlp_diel_stern = laplace.double_layer(
        dirichl_space_diel, dirichl_space_stern, dirichl_space_stern, assembler=operator_assembler
    ) #diel to stern
    identity_stern = sparse.identity(dirichl_space_stern, dirichl_space_stern, dirichl_space_stern)
    slp_in_stern = laplace.single_layer(
        neumann_space_stern, dirichl_space_stern, dirichl_space_stern, assembler=operator_assembler
    )
    dlp_in_stern = laplace.double_layer(
        dirichl_space_stern, dirichl_space_stern, dirichl_space_stern, assembler=operator_assembler
    )
    slp_out_stern = modified_helmholtz.single_layer(
        neumann_space_stern, dirichl_space_stern, dirichl_space_stern, kappa, assembler=operator_assembler
    )
    dlp_out_stern = modified_helmholtz.double_layer(
        dirichl_space_stern, dirichl_space_stern, dirichl_space_stern, kappa, assembler=operator_assembler
    )

    A = bempp.api.BlockedOperator(4, 4)

    A[0, 0] = 0.5 * identity_diel + dlp_in_diel
    A[0, 1] = -slp_in_diel
    A[1, 0] = 0.5 * identity_diel - dlp_out_diel
    A[1, 1] = (ep_in / ep_stern) * slp_out_diel
    A[1, 2] = dlp_stern_diel
    A[1, 3] = -slp_diel_stern
    A[2, 0] = -dlp_stern_diel
    A[2, 1] = (ep_in / ep_stern) * slp_diel_stern
    A[2, 2] = 0.5 * identity_stern + dlp_in_stern
    A[2, 3] = -slp_in_stern
    A[3, 2] = 0.5 * identity_stern - dlp_out_stern
    A[3, 3] = (ep_stern / ep_out) * slp_out_stern

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
        rhs_2 = bempp.api.GridFunction(neumann_space_diel, coefficients=coefs_neumann_diel)
        rhs_3 = bempp.api.GridFunction(dirichl_space_stern, coefficients=coefs_dirichl_stern)
        rhs_4 = bempp.api.GridFunction(neumann_space_stern, coefficients=coefs_neumann_stern)

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

    self.rhs["rhs_1"], self.rhs["rhs_2"], self.rhs["rhs_3"], self.rhs["rhs_4"] = rhs_1, rhs_2, rhs_3, rhs_4


def block_diagonal_preconditioner(solute):
    from scipy.sparse import diags, bmat
    from scipy.sparse.linalg import aslinearoperator
    from pbj.electrostatics.solute import matrix_to_discrete_form, rhs_to_discrete_form

    matrix_A = solute.matrices["A"]

    block1 = matrix_A[0, 0]
    block2 = matrix_A[0, 1]
    block3 = matrix_A[1, 0]
    block4 = matrix_A[1, 1]

    diag11 = (
        block1._op1._alpha * block1._op1._op.weak_form().to_sparse().diagonal()
        + block1._op2.descriptor.singular_part.weak_form().to_sparse().diagonal()
    )
    diag12 = (
        block2._alpha
        * block2._op.descriptor.singular_part.weak_form().to_sparse().diagonal()
    )
    diag21 = (
        block3._op1._alpha * block3._op1._op.weak_form().to_sparse().diagonal()
        + block3._op2._alpha
        * block3._op2._op.descriptor.singular_part.weak_form().to_sparse().diagonal()
    )
    diag22 = (
        block4._alpha
        * block4._op.descriptor.singular_part.weak_form().to_sparse().diagonal()
    )

    d_aux = 1 / (diag22 - diag21 * diag12 / diag11)
    diag11_inv = 1 / diag11 + 1 / diag11 * diag12 * d_aux * diag21 / diag11
    diag12_inv = -1 / diag11 * diag12 * d_aux
    diag21_inv = -d_aux * diag21 / diag11
    diag22_inv = d_aux

    block_mat_precond = bmat(
        [[diags(diag11_inv), diags(diag12_inv)], [diags(diag21_inv), diags(diag22_inv)]]
    ).tocsr()

    solute.matrices["preconditioning_matrix_gmres"] = aslinearoperator(
        block_mat_precond
    )
    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]

    solute.matrices["A_discrete"] = matrix_to_discrete_form(
        solute.matrices["A_final"], "weak"
    )
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(
        solute.rhs["rhs_final"], "weak", solute.matrices["A"]
    )

    """
    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    identity_diag = identity.weak_form().to_sparse().diagonal()
    slp_in_diag = laplace.single_layer(neumann_space, dirichl_space, dirichl_space,
                                       assembler="only_diagonal_part").weak_form().get_diagonal()
    dlp_in_diag = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space,
                                       assembler="only_diagonal_part").weak_form().get_diagonal()
    slp_out_diag = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                                   assembler="only_diagonal_part").weak_form().get_diagonal()
    dlp_out_diag = modified_helmholtz.double_layer(neumann_space, dirichl_space, dirichl_space, kappa,
                                                   assembler="only_diagonal_part").weak_form().get_diagonal()

    #if permuted_rows:
    diag11 = .5 * identity_diag - dlp_out_diag
    diag12 = (ep_in / ep_ex) * slp_out_diag
    diag21 = .5 * identity_diag + dlp_in_diag
    diag22 = -slp_in_diag
    """


def mass_matrix_preconditioner(solute):
    from pbj.electrostatics.solute import matrix_to_discrete_form, rhs_to_discrete_form

    # Option A:
    """
    from bempp.api.utils.helpers import get_inverse_mass_matrix
    from bempp.api.assembly.blocked_operator import BlockedDiscreteOperator

    matrix = solute.matrices["A"]
    nrows = len(matrix.range_spaces)
    range_ops = np.empty((nrows, nrows), dtype="O")

    for index in range(nrows):
        range_ops[index, index] = get_inverse_mass_matrix(matrix.range_spaces[index],
                                                          matrix.dual_to_range_spaces[index])

    preconditioner = BlockedDiscreteOperator(range_ops)
    solute.matrices['preconditioning_matrix_gmres'] = preconditioner
    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]
    solute.matrices["A_discrete"] = matrix_to_discrete_form(solute.matrices["A_final"], "weak")
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(solute.rhs["rhs_final"], "weak", solute.matrices["A"])

    """
    solute.matrices["A_final"] = solute.matrices["A"]
    solute.rhs["rhs_final"] = [solute.rhs["rhs_1"], solute.rhs["rhs_2"]]
    solute.matrices["A_discrete"] = matrix_to_discrete_form(
        solute.matrices["A_final"], "strong"
    )
    solute.rhs["rhs_discrete"] = rhs_to_discrete_form(
        solute.rhs["rhs_final"], "strong", solute.matrices["A"]
    )