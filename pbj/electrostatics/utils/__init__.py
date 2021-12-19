from .analytical import *

def solver(A, rhs, tolerance, restart_value, max_iterations, precond=None):
    from scipy.sparse.linalg import gmres
    from bempp.api.linalg.iterative_solvers import IterationCounter

    callback = IterationCounter(True)

    if precond is None:
        x, info = gmres(
            A,
            rhs,
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
            tol=tolerance,
            restart=restart_value,
            maxiter=max_iterations,
            callback=callback,
        )

    return x, info, callback.count
