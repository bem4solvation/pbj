r"""Müller internal formulation for the Poisson-Boltzmann equation.

This module implements the Müller internal formulation for solving the Poisson-Boltzmann
(PB) equation using boundary element methods (BEM). This formulation combines interior
Laplace and exterior modified Helmholtz operators in a way that emphasizes interior
region treatment.

The Müller internal method is particularly suited for systems where accurate interior
field representation is critical.

Functions:
    verify_parameters: Validates formulation parameters (no parameters required).
    lhs: Assembles the left-hand side system matrix using boundary element operators.
    rhs: Constructs the right-hand side vector from charge distributions.
    mass_matrix_preconditioner: Builds a mass-matrix based GMRES preconditioner.
    calculate_potential: Solves the system and computes the potential.

Module Attributes:
    invert_potential (bool): Flag indicating whether potential inversion should be
        applied. Set to False for this formulation.
"""

import numpy as np
import bempp_cl as bempp
import bempp_cl.api
from bempp_cl.api.operators.boundary import sparse, laplace, modified_helmholtz
from .common import calculate_potential_one_surface

invert_potential = False


def verify_parameters(self):
    r"""Verifies that the Poisson-Boltzmann formulation parameters are valid.

    This formulation has no special parameters to verify beyond basic configuration.

    Args:
        self (Solute): The instance of the Solute class.

    Returns:
        bool: Always returns True as no parameters need validation.
    """
    return True


def lhs(self):
    r"""Assembles the left-hand side system matrix for the Müller internal formulation.

    Constructs the discrete boundary element system matrix by combining interior
    Laplace and exterior modified Helmholtz operators with identity scaling.
    This formulation emphasizes interior region contributions.

    Args:
        self (Solute): The Solute instance containing geometry, spaces, and parameters.

    Notes:
        - Stores the main system matrix in self.matrices['A']
        - Uses direct combination of operators without parameter scaling
        - Suitable for problems requiring accurate interior field representation
    """
    dirichl_space = self.dirichl_space
    neumann_space = self.neumann_space
    ep_in = self.ep_in
    ep_ex = self.ep_ex
    kappa = self.kappa
    operator_assembler = self.operator_assembler

    dlp_in = laplace.double_layer(
        dirichl_space, dirichl_space, dirichl_space, assembler=operator_assembler
    )
    slp_in = laplace.single_layer(
        neumann_space, dirichl_space, dirichl_space, assembler=operator_assembler
    )
    hlp_in = laplace.hypersingular(
        dirichl_space, neumann_space, neumann_space, assembler=operator_assembler
    )
    adlp_in = laplace.adjoint_double_layer(
        neumann_space, neumann_space, neumann_space, assembler=operator_assembler
    )

    dlp_ex = modified_helmholtz.double_layer(
        dirichl_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler
    )
    slp_ex = modified_helmholtz.single_layer(
        neumann_space, dirichl_space, dirichl_space, kappa, assembler=operator_assembler
    )
    hlp_ex = modified_helmholtz.hypersingular(
        dirichl_space, neumann_space, neumann_space, kappa, assembler=operator_assembler
    )
    adlp_ex = modified_helmholtz.adjoint_double_layer(
        neumann_space, neumann_space, neumann_space, kappa, assembler=operator_assembler
    )

    phi_identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_identity = sparse.identity(neumann_space, neumann_space, neumann_space)

    ep = ep_ex / ep_in

    A = bempp.api.BlockedOperator(2, 2)
    A[0, 0] = phi_identity + dlp_in - dlp_ex
    A[0, 1] = -slp_in + ((1.0 / ep) * slp_ex)
    A[1, 0] = -hlp_in + (ep * hlp_ex)
    A[1, 1] = dph_identity - adlp_in + adlp_ex

    self.matrices["A"] = A


def rhs(self):
    r"""Constructs the right-hand side vector from the charge distribution.

    Computes boundary element grid functions representing the potential and its normal
    derivative due to the solute charges using direct Green's function evaluation.

    Args:
        self (Solute): The Solute instance containing charges, geometry, and parameters.

    Notes:
        - Stores RHS components in self.rhs:
          - 'rhs_1': Potential due to charge distribution
          - 'rhs_2': Normal derivative of potential
        - Uses direct Green's function computation (no FMM acceleration)
    """
    dirichl_space = self.dirichl_space
    q = self.q
    x_q = self.x_q
    ep_in = self.ep_in

    @bempp.api.real_callable
    def d_green_func(x, n, domain_index, result):
        nrm = np.sqrt(
            (x[0] - x_q[:, 0]) ** 2 + (x[1] - x_q[:, 1]) ** 2 + (x[2] - x_q[:, 2]) ** 2
        )
        const = -1.0 / (4.0 * np.pi * ep_in)
        result[:] = const * np.sum(q * np.dot(x - x_q, n) / (nrm**3))

    @bempp.api.real_callable
    def green_func(x, n, domain_index, result):
        nrm = np.sqrt(
            (x[0] - x_q[:, 0]) ** 2 + (x[1] - x_q[:, 1]) ** 2 + (x[2] - x_q[:, 2]) ** 2
        )
        result[:] = np.sum(q / nrm) / (4.0 * np.pi * ep_in)

    rhs_1 = bempp.api.GridFunction(dirichl_space, fun=green_func)
    rhs_2 = bempp.api.GridFunction(dirichl_space, fun=d_green_func)

    self.rhs["rhs_1"] = rhs_1
    self.rhs["rhs_2"] = rhs_2


def mass_matrix_preconditioner(solute):
    r"""Builds a mass-matrix based preconditioner for GMRES acceleration.

    Constructs a preconditioner using the mass matrix from the boundary element
    discretization. This provides a simple approach for improving GMRES convergence.

    Args:
        solute (Solute): The Solute instance containing operators and parameters.

    Notes:
        - Converts the system to discrete form using 'strong' form formulation
        - Converts RHS to discrete form with rhs_to_discrete_form()
        - Does not apply explicit inverse mass matrix (uses identity approximation)
    """
    from pbj.implicit_solvent.utils import matrix_to_discrete_form, rhs_to_discrete_form

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


def calculate_potential(self, rerun_all, rerun_rhs):
    r"""Solves the linear system and computes the electrostatic potential.

    Orchestrates the solution workflow: assembles or reuses the linear system,
    solves using iterative methods (GMRES), and extracts the potential and field
    values on the boundary and throughout the domain.

    Args:
        self (Solute): The Solute instance containing matrices, RHS, and solver parameters.
        rerun_all (bool): If True, recompute the full linear system from scratch.
        rerun_rhs (bool): If True, recompute only the right-hand side vector.

    Notes:
        - Stores computed results in self.results dictionary
        - Updates potential values (phi) on the surface and in the domain
    """
    calculate_potential_one_surface(self, rerun_all, rerun_rhs)
