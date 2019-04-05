#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import argparse
import logging


logging.basicConfig(level=logging.INFO)

np.set_printoptions(precision=3)


def _parse_args():
    _description = 'Simulate two correlated random variables. Compute sample covariance matrix and PCA'
    parser = argparse.ArgumentParser(description=_description)
    parser.add_argument('-o', '--observations', type=int, default=100, help='number of observations')
    parser.add_argument('-c','--correlation', type=float, default=0.5, help='correlation between the two variables')
    parser.add_argument('-v', '--volatility', nargs=2, type=float, default=[0.2, 0.2],
                        help='volatility of the two variables')

    args = parser.parse_args()

    return args


def _compute_sample_vector(observations, volatility_vector, correlation):
    std_norm_vector = np.random.randn(observations, 2)
    correlated_array = std_norm_vector[:, 0] * correlation + np.sqrt(1 - correlation**2) * std_norm_vector[:, 1]
    correlated_std_norm = np.column_stack((std_norm_vector[:, 0], correlated_array))
    res = np.multiply(correlated_std_norm, volatility_vector)
    return res


def main():
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


if __name__ == '__main__':
    main()

