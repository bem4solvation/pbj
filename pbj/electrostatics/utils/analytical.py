import numpy
from numpy import pi
from scipy import special
from scipy.special import factorial


def get_K(x, n):
    """
    It computes the polinomials K needed for Kirkwood-1934 solutions.
    K_n(x) in Equation 4 in Kirkwood 1934.
    Arguments
    ----------
    x: float, evaluation point of K.
    n: int, number of terms desired in the expansion.
    Returns
    --------
    K: float, polinomials K.
    """

    K = 0.0
    n_fact = factorial(n)
    n_fact2 = factorial(2 * n)
    for s in range(n + 1):
        K += (
            2**s
            * n_fact
            * factorial(2 * n - s)
            / (factorial(s) * n_fact2 * factorial(n - s))
            * x**s
        )

    return K


def an_P(q, xq, E_1, E_2, R, kappa, a, N):
    """
    It computes the solvation energy according to Kirkwood-1934.
    Arguments
    ----------
    q    : array, charges.
    xq   : array, positions of the charges.
    E_1  : float, dielectric constant inside the sphere.
    E_2  : float, dielectric constant outside the sphere.
    R    : float, radius of the sphere.
    kappa: float, reciprocal of Debye length.
    a    : float, radius of the Stern Layer.
    N    : int, number of terms desired in the polinomial expansion.
    Returns
    --------
    E_P  : float, solvation energy.
    """

    qe = 1.60217646e-19
    Na = 6.0221415e23
    E_0 = 8.854187818e-12
    cal2J = 4.184

    PHI = numpy.zeros(len(q))
    for K in range(len(q)):
        rho = numpy.sqrt(numpy.sum(xq[K] ** 2))
        zenit = numpy.arccos(xq[K, 2] / rho)
        azim = numpy.arctan2(xq[K, 1], xq[K, 0])

        phi = 0.0 + 0.0 * 1j
        for n in range(N):
            for m in range(-n, n + 1):
                P1 = special.lpmv(numpy.abs(m), n, numpy.cos(zenit))

                Enm = 0.0
                for k in range(len(q)):
                    rho_k = numpy.sqrt(numpy.sum(xq[k] ** 2))
                    zenit_k = numpy.arccos(xq[k, 2] / rho_k)
                    azim_k = numpy.arctan2(xq[k, 1], xq[k, 0])
                    P2 = special.lpmv(numpy.abs(m), n, numpy.cos(zenit_k))

                    Enm += (
                        q[k]
                        * rho_k**n
                        * factorial(n - numpy.abs(m))
                        / factorial(n + numpy.abs(m))
                        * P2
                        * numpy.exp(-1j * m * azim_k)
                    )

                C2 = (
                    (kappa * a) ** 2
                    * get_K(kappa * a, n - 1)
                    / (
                        get_K(kappa * a, n + 1)
                        + n
                        * (E_2 - E_1)
                        / ((n + 1) * E_2 + n * E_1)
                        * (R / a) ** (2 * n + 1)
                        * (kappa * a) ** 2
                        * get_K(kappa * a, n - 1)
                        / ((2 * n - 1) * (2 * n + 1))
                    )
                )
                C1 = (
                    Enm
                    / (E_2 * E_0 * a ** (2 * n + 1))
                    * (2 * n + 1)
                    / (2 * n - 1)
                    * (E_2 / ((n + 1) * E_2 + n * E_1)) ** 2
                )

                if n == 0 and m == 0:
                    Bnm = Enm / (E_0 * R) * (1 / E_2 - 1 / E_1) - Enm * kappa * a / (
                        E_0 * E_2 * a * (1 + kappa * a)
                    )
                else:
                    Bnm = (
                        1.0
                        / (E_1 * E_0 * R ** (2 * n + 1))
                        * (E_1 - E_2)
                        * (n + 1)
                        / (E_1 * n + E_2 * (n + 1))
                        * Enm
                        - C1 * C2
                    )

                phi += Bnm * rho**n * P1 * numpy.exp(1j * m * azim)

        PHI[K] = numpy.real(phi) / (4 * pi)

    C0 = qe**2 * Na * 1e-3 * 1e10 / (cal2J)
    E_P = 0.5 * C0 * numpy.sum(q * PHI)

    return E_P
