#!/usr/bin/env python

import argparse
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def parse_args():
    """parse arguments:
    sample_size = number of observations per sample (int)
    num_samples = number of total samples to take (int)
    corr = population correlation [-1.0, 1.0] (float)
    """
    _parser = argparse.ArgumentParser(description='simulate correlated normal'
                                      'variables')

    _parser.add_argument('sample_size', type=int, help='observations per sample (int)')
    _parser.add_argument('num_samples', type=int, help='total number of samples (int)')
    _parser.add_argument('corr', type=float, help='population correlation [-1, 1] (float)')
    _parsed_args = _parser.parse_args()
    return _parsed_args


def main(_sample_size, _num_samples, _corr=0):
    _num_rows = _sample_size * _num_samples
    _rand = np.random.randn(_num_rows, 2)
    _corr_rand = _corr * _rand[:, 0] + sqrt(1-_corr**2)*_rand[:, 1]
    _rand = list(zip(_rand[:, 0], _corr_rand))
    _sample_idx = list(range(0, len(_rand)+_sample_size, _sample_size))
    _corr_est = [np.corrcoef(_rand[_sample_idx[i]:_sample_idx[i+1]], rowvar=0)[0][1]
                 for i in range(len(_sample_idx)-1)]
    _total_corr = np.corrcoef(_rand, rowvar=0)[0][1]
    logging.info("number of samples: {}".format(len(_corr_est)))
    plt.hist(_corr_est, bins=30, range=(-1.0, 1.0))
    plt.axvline(_corr, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(_total_corr, color='r', linestyle='dashed', linewidth=1)

    _title = "Correlation: {} Sample Size: {} Samples: {}".format(_corr, _sample_size, _num_samples)
    plt.title(_title)
    plt.show()


if __name__ == '__main__':
    _args = parse_args()
    _sample_size = _args.sample_size
    _num_samples = _args.num_samples
    _corr = _args.corr
    main(_sample_size, _num_samples, _corr)

