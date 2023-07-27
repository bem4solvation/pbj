from .analytical import *


def solver(A, rhs, tolerance, restart_value, max_iterations, initial_guess=None, precond=None):
    from scipy.sparse.linalg import gmres
    from bempp.api.linalg.iterative_solvers import IterationCounter

    callback = IterationCounter(True)

    if precond is None:
        x, info = gmres(
            A,
            rhs,
            x0 = initial_guess,
            tol=tolerance,
            restart=restart_value,
            maxiter=max_iterations,
            callback=callback,
        )
    else:
        x, info = gmres(
            A,
            rhs,
            M=precond,
            x0 = initial_guess,
            tol=tolerance,
            restart=restart_value,
            maxiter=max_iterations,
            callback=callback,
        )

    return x, info, callback.count


def matrix_to_discrete_form(matrix, discrete_form_type):
    if discrete_form_type == "strong":
        matrix_discrete = matrix.strong_form()
    elif discrete_form_type == "weak":
        matrix_discrete = matrix.weak_form()
    else:
        raise ValueError("Unexpected discrete type: %s" % discrete_form_type)

    return matrix_discrete


def rhs_to_discrete_form(rhs_list, discrete_form_type, A):
    from bempp.api.assembly.blocked_operator import (
        coefficients_from_grid_functions_list,
        projections_from_grid_functions_list,
    )

    if discrete_form_type == "strong":
        rhs = coefficients_from_grid_functions_list(rhs_list)
    elif discrete_form_type == "weak":
        rhs = projections_from_grid_functions_list(rhs_list, A.dual_to_range_spaces)
    else:
        raise ValueError("Unexpected discrete form: %s" % discrete_form_type)

    return rhs
