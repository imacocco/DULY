import numpy as np
from scipy.optimize import minimize

import dadapy.utils_.utils as ut


def ML_fun_gPAk(params, args):
    """
    The function returns the log-Likelihood expression to be minimized.

    Requirements:

    * **params**: array of initial values for ''a'', ''b''
    * **args**: additional parameters ''kopt'', ''Vi'' entering the Likelihood

    Note:

    * **b**: correspond to the ''log(rho)'', as in Eq. (S1)
    * **a**: the linear correction, as in Eq. (S1)

    """

    Fi = params[0]

    a = params[1]

    kopt = args[0]

    vij = args[1]

    grads_ij = args[2]

    gb = kopt

    ga = np.sum(grads_ij)

    L0 = Fi * gb + a * ga

    for j in range(kopt):
        t = Fi + a * grads_ij[j]

        s = np.exp(t)

        tt = vij[j] * s

        L0 = L0 - tt

    return -L0


def ML_fun_gpPAk(params, args):
    """
    The function returns the log-Likelihood expression to be minimized.

    Requirements:

    * **params**: array of initial values for ''a'', ''b''
    * **args**: additional parameters ''kopt'', ''Vi'' entering the Likelihood

    Note:

    * **b**: correspond to the ''log(rho)'', as in Eq. (S1)
    * **a**: the linear correction, as in Eq. (S1)

    """

    Fi = params[0]

    a = params[1]

    kopt = args[0]

    vij = args[1]

    grads_ij = args[2]

    gb = kopt

    ga = (kopt + 1) * kopt * 0.5

    L0 = Fi * gb + np.sum(grads_ij) + a * ga

    for j in range(kopt):
        jf = float(j + 1)
        t = Fi + grads_ij[j] + a * jf

        s = np.exp(t)

        tt = vij[j] * s

        L0 = L0 - tt

    return -L0


def ML_fun(params, args):
    """
    The function returns the log-Likelihood expression to be minimized.

    Requirements:

    * **params**: array of initial values for ''a'', ''b''
    * **args**: additional parameters ''kopt'', ''Vi'' entering the Likelihood

    Note:

    * **b**: correspond to the ''log(rho)'', as in Eq. (S1)
    * **a**: the linear correction, as in Eq. (S1)

    """
    # g = [0, 0]
    b = params[0]
    a = params[1]
    kopt = args[0]
    gb = kopt
    ga = (kopt + 1) * kopt * 0.5
    L0 = b * gb + a * ga
    Vi = args[1]
    for k in range(1, kopt):
        jf = float(k)
        t = b + a * jf
        s = np.exp(t)
        tt = Vi[k - 1] * s
        L0 = L0 - tt
    return -L0


def ML_hess_fun(params, args):
    """
    The function returns the expressions for the asymptotic variances of the estimated parameters.

    Requirements:

    * **params**: array of initial values for ''a'', ''b''
    * **args**: additional parameters ''kopt'', ''Vi'' entering the Likelihood

    Note:

    * **b**: correspond to the ''log(rho)'', as in Eq. (S1)
    * **a**: the linear correction, as in Eq. (S1)

    """
    g = [0, 0]
    b = params[0]
    a = params[1]
    kopt = args[0]
    gb = kopt
    ga = (kopt + 1) * kopt * 0.5
    L0 = b * gb + a * ga
    Vi = args[1]
    Cov2 = np.array([[0.0] * 2] * 2)
    for k in range(1, kopt):
        jf = float(k)
        t = b + a * jf
        s = np.exp(t)
        tt = Vi[k - 1] * s
        L0 = L0 - tt
        gb = gb - tt
        ga = ga - jf * tt
        Cov2[0][0] = Cov2[0][0] - tt
        Cov2[0][1] = Cov2[0][1] - jf * tt
        Cov2[1][1] = Cov2[1][1] - jf * jf * tt
    Cov2[1][0] = Cov2[0][1]
    Cov2 = Cov2 * (-1)
    Covinv2 = np.linalg.inv(Cov2)

    g[0] = np.sqrt(Covinv2[0][0])
    g[1] = np.sqrt(Covinv2[1][1])
    return g


def MLmax(rr, kopt, Vi):
    """
    This function uses the scipy.optimize package to minimize the function returned by ''ML_fun'', and
    the ''ML_hess_fun'' for the analytical calculation of the Hessian for errors estimation.
    It returns the value of the density which minimize the log-Likelihood in Eq. (S1)

    Requirements:

    * **rr**: is the initial value for the density, by using the standard k-NN density estimator
    * **kopt**: is the optimal neighborhood size k as return by the Likelihood Ratio test
    * **Vi**: is the list of the ''kopt'' volumes of the shells defined by two successive nearest neighbors of the current point

    #"""
    # results = minimize(ML_fun, [rr, 0.], method='Nelder-Mead', args=([kopt, Vi],),
    #                    options={'maxiter': 1000})

    results = minimize(
        ML_fun,
        [rr, 0.0],
        method="Nelder-Mead",
        tol=1e-6,
        args=([kopt, Vi]),
        options={"maxiter": 1000},
    )

    # err = ML_hess_fun(results.x, [kopt, Vi])
    # a_err = err[1]
    rr = results.x[0]  # b
    print(results.message)
    return rr


def MLmax_gPAk(rr, kopt, Vi, grads_ij):
    results = minimize(
        ML_fun_gPAk,
        [rr, 0.0],
        method="Nelder-Mead",
        tol=1e-6,
        args=([kopt, Vi, grads_ij]),
        options={"maxiter": 1000},
    )

    rr = results.x[0]  # b
    print(results.message)
    return rr


def MLmax_gpPAk(rr, kopt, Vi, grads_ij):
    results = minimize(
        ML_fun_gpPAk,
        [rr, 0.0],
        method="Nelder-Mead",
        tol=1e-6,
        args=([kopt, Vi, grads_ij]),
        options={"maxiter": 1000},
    )

    rr = results.x[0]  # b
    print(results.message)
    return rr


def MLmax_kNN_corr(Fis, kstar, Vis, dist_indices, Fij_list, Fij_var_list, alpha):
    print("ML maximisation started")

    # methods: 'Nelder-Mead', 'BFGS'
    # results = minimize(ML_fun_kNN_corr, Fis, method='Nelder-Mead', tol=1e-6,
    #                    args=([kstar, Vis, dist_indices, Fij_list, Fij_var_list, alpha]),
    #                    options={'maxiter': 50000})

    results = minimize(
        ML_fun_kNN_corr,
        Fis,
        method="CG",
        tol=1e-6,
        jac=ML_fun_grad,
        args=([kstar, Vis, dist_indices, Fij_list, Fij_var_list, alpha]),
        options={"maxiter": 100},
    )

    rr = results.x  # b
    print(results.message)
    print(results.nit)
    print(results.nfev)
    print(results.njev)
    print(np.mean(abs(results.jac)))
    return rr


if __name__ == "__main__":
    pass
