#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import argparse
import logging
import pdb

logging.basicConfig(level=logging.INFO)


def _parse_args():
    _description = 'Simulate two correlated random variables. Compute sample covariance matrix and PCA'
    parser = argparse.ArgumentParser(description=_description)
    parser.add_argument('-o', '--observations', type=int, default=100, help='number of observations')
    parser.add_argument('-c','--correlation', type=float, default=0.5, help='correlation between the two variables')
    parser.add_argument('-v', '--volatility', type=list, default=[0.2, 0.2], help='volatility of the two variables')

    args = parser.parse_args()

    return args


def _compute_sample_vector(observations, volatility_vector, correlation):
    std_norm_vector = np.random.randn(observations, 2)
    correlated_array = std_norm_vector[:, 0] * correlation + np.sqrt(1 - correlation**2) * std_norm_vector[:, 1]
    correlated_std_norm = list(zip(std_norm_vector[:, 0], correlated_array))
    res = correlated_std_norm * volatility_vector
    return res


def main():
    args = _parse_args()

    correlated_rvs = _compute_sample_vector(args.observations, args.volatility, args.correlation)
    sample_covariance = np.cov(correlated_rvs, rowvar=False)
    sample_correlation = np.corrcoef(correlated_rvs, rowvar=False)
    logging.info(sample_covariance)
    logging.info(sample_correlation)


if __name__ == '__main__':
    main()

