#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""example script demonstrating usage of scipy.optimize.minimize"""

from scipy.optimize import minimize, LinearConstraint
import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


def f_oned_quadratic(x, *args):
    """function to be optimized

    Args:
        x (float): function input
        args: coefficients of polynomial args[0] * x**2 + args[1] * x + args[2]

    Returns:
        float
    """
    return args[0] * x**2 + args[1] * x + args[2]


def one_dimension_example():
    x = np.arange(-10, 10, 0.1)
    args = (5, -20, 0)
    y = f_oned_quadratic(x, *args)
    res = minimize(f_oned_quadratic, x[0], args=args)
    analytic_res = -args[1] / (2 * args[0])
    logging.info('minimum occurs at {}'.format(res.x))
    logging.info('analytical minimum occurs at {}'.format(analytic_res))
    plt.axvline(res.x, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(analytic_res, color='r', linestyle='dashed', linewidth=1)
    plt.plot(x, y)
    plt.show()


def utility_fn(weights, *args):
    """
    Utility = alpha - lambda * variance
    :param weights: np.array (n, 1) weights of positions in portfolio
    :param args: alpha, covariance_mat, port_lambda
    :return: float
    """
    alpha, covariance_mat, port_lambda = args
    port_alpha = alpha.T @ weights
    port_variance = weights.T @ covariance_mat @ weights

    return float(port_alpha - port_lambda * port_variance)


def utility_example():
    x = np.arange(-10, 10, 1)
    y = np.arange(-10, 10, 1)
    x0 = np.array([1, -1])

    alpha = np.zeros((2, 1))
    alpha = np.array([4, -2])
    alpha = alpha.reshape((2, 1))

    beta = np.array([2, 1])
    beta = beta.reshape((2, 1))
    sigma = 0.3
    # covar = sigma * beta @ beta.T * sigma
    covar = np.array([[0.4, 0], [0, 0.4]])

    port_lambda = 10
    z = []
    for y_val in y:
        z.append([])
        last_row = len(z) - 1
        for x_val in x:
            inputs = np.array([x_val, y_val])
            inputs = inputs.reshape(2, 1)
            val = utility_fn(inputs, alpha, covar, port_lambda)
            z[last_row].append(val)
    constraint = LinearConstraint(np.array([1, 1]), -np.inf, np.array(1))
    nm_res = minimize(lambda weights, *args: -1*utility_fn(weights, *args), x0, args=(alpha, covar, port_lambda),
                      method='Nelder-Mead', constraints=constraint)
    logging.info('Nelder-Mead result is {}'.format(nm_res.x))
    logging.warning('Nelder-Mead ignores constraints')

    default_res = minimize(lambda weights, *args: -1*utility_fn(weights, *args), x0, args=(alpha, covar, port_lambda),
                           constraints=constraint)

    logging.info('Default result is {}'.format(default_res.x))
    z = np.array(z)
    plt.contourf(x, y, z, 10)
    plt.show()


def main():

    utility_example()


if __name__ == '__main__':
    main()
