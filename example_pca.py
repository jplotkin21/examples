#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import argparse
import logging
import sklearn.decomposition
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

np.set_printoptions(precision=3)


def _parse_args():
    """parse command line arguments

    Args:
        :none

    Returns:
        namespace: (observations, correlation, volatility)
    """
    _description = 'Simulate two correlated random variables. Compute sample covariance matrix and PCA'
    parser = argparse.ArgumentParser(description=_description)
    parser.add_argument('-o', '--observations', type=int, default=100, help='number of observations')
    parser.add_argument('-c', '--correlation', type=float, default=0.5, help='correlation between the two variables')
    parser.add_argument('-v', '--volatility', nargs=2, type=float, default=[0.2, 0.2],
                        help='volatility of the two variables')

    args = parser.parse_args()

    return args


def _compute_sample_vector(observations, volatility_vector, correlation):
    """compute a matrix of correlated random variables

    Args:
        observations(int): number of observations to generate
        volatility_vector(list): volatility of each of the two random variables
        correlation(float): correlation coefficient between the two random variables

    Returns:
          np.matrix: (observations, 2) matrix of correlated random variables
    """
    std_norm_vector = np.random.randn(observations, 2)
    correlated_array = std_norm_vector[:, 0] * correlation + np.sqrt(1 - correlation**2) * std_norm_vector[:, 1]
    correlated_std_norm = np.column_stack((std_norm_vector[:, 0], correlated_array))
    res = np.multiply(correlated_std_norm, volatility_vector)
    return res


def main():
    """main entry point for function. compute sample covariance matrix and pca values

    Args:
        :none

    Returns:
        None
    """
    args = _parse_args()
    corr = args.correlation
    vol_array = args.volatility
    logging.info('volatility vector: {}, correlation: {}'.format(vol_array, corr))
    theoretical_covariance = np.matrix([[vol_array[0]**2, vol_array[0]*vol_array[1]*corr],
                                       [vol_array[0]*vol_array[1]*corr, vol_array[1]**2]])
    logging.info('theoretical covariance:\n{}'.format(theoretical_covariance))
    correlated_rvs = _compute_sample_vector(args.observations, args.volatility, args.correlation)
    sample_covariance = np.cov(correlated_rvs, rowvar=False)
    sample_correlation = np.corrcoef(correlated_rvs, rowvar=False)
    logging.info('sample covariance:\n{}'.format(sample_covariance))
    logging.info('sample correlation:\n{}'.format(sample_correlation))

    pca = sklearn.decomposition.PCA()
    pca.fit_transform(correlated_rvs)
    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_

    logging.info('eigenvectors of PCA:\n{}'.format(eigenvectors))
    logging.info('eigenvalues of PCA: {}'.format(eigenvalues))

    np_eigenvalues, np_eigenvectors = np.linalg.eig(sample_covariance)
    explained_var_ratio = [v / sum(np_eigenvalues) for v in np_eigenvalues]
    scaled_eigenvectors = np.multiply(np_eigenvectors, explained_var_ratio)
    logging.info('eigenvectors of covariance matrix:\n{}'.format(np_eigenvectors))
    logging.info('eigenvalues of covariance matrix: {}'.format(np_eigenvalues))
    logging.info('explained var ratio: {}'.format(explained_var_ratio))

    fig = plt.figure()
    ax = fig.add_subplot('111')
    ax.scatter(correlated_rvs[:, 0], correlated_rvs[:, 1], s=20)
    ax.plot([0, np_eigenvectors[0, 0]], [0, np_eigenvectors[1, 0]], color='b')
    ax.plot([0, np_eigenvectors[0, 1]], [0, np_eigenvectors[1, 1]], color='b')
    ax.plot([0, scaled_eigenvectors[0, 0]], [0, scaled_eigenvectors[1, 0]], color='g')
    ax.plot([0, scaled_eigenvectors[0, 1]], [0, scaled_eigenvectors[1, 1]], color='g')
    ax.set_title('Volatility Vector: {}, Correlation: {}\nNumber Observations: {}'.format(
        vol_array, corr, args.observations)
    )
    max_val = np.max(np.array(list(map(abs, correlated_rvs))))
    max_val = max(max_val, 1)
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

