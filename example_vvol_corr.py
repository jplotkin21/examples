#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


@np.vectorize
def brownian_correlation(t1, t2):
    """correlation between two brownian motions
    let Y(t1) = W(t1), Y(t2) = W(t2) then
    covar(Y(t1),Y(t2)) = min(t1, t2)
    Var(Y(t1)) = t1, Var(Y(t2)) = t2
    and corr(Y(t1),Y(t2)) = sqrt(min(t1,t2)/max(t1,t2))

    Args:
        t1 (float): expiry time of term 1
        t2 (float): expiry time of term 2

    Returns:
         Float: correlation of the two terms
    """
    res = math.sqrt(min(t1, t2)/max(t1, t2))

    return res


@np.vectorize
def integral_brownian_correlation(t1, t2):
    """correlation between the integral of two brownian motions
    let Y(t1) = integral[0, t1](W(s)ds),
    Y(t2) = integral[0, t2](W(s)ds) then
    covar(Y(t1),Y(t2)) = min(t1, t2)^2*(0.5*max(t1, t2) - 1/6*min(t1, t2))
    var(Y(t1)) = t^3/3
    corr(Y(t1), Y(t2)) =
    3*min(t1, t2)^2*(0.5*max(t1, t2) - min(t1, t2)/6)/(t1^(3/2)*t2^(3/2))

    Args:
        t1 (float): expiry time of term 1
        t2 (float): expiry time of term 2

    Returns:
         Float: correlation of the two terms
    """

    covar = min(t1, t2)**2*(0.5*max(t1, t2) - min(t1, t2)/6)
    res = 3*covar/(t1**(3/2)*t2**(3/2))
    return res


if __name__ == '__main__':
    _t2 = 1
    _t1 = np.arange(0.01, 1, 0.01)
#    _brown_corr = [brownian_correlation(t, _t2) for t in _t1]
#    _int_corr = [integral_brownian_correlation(t, _t2) for t in _t1]
    _brown_corr = brownian_correlation(_t1, _t2)
    _int_corr = integral_brownian_correlation(_t1, _t2)
    _fig = plt.figure()
    _ax = _fig.add_subplot(111)
    _ax.plot(_t1, _brown_corr, label="brownian")
    _ax.plot(_t1, _int_corr, label="integral_brownian")
    title = ("Correlation\nBrownian={W(t1), W(t2)}\nIntegral Brownian="
             "{int[0, t1](W(s)ds), int[0, t2](W(s)ds)}")
    _ax.set_title(title)
    _ax.legend()
    _ax.set_xlabel("t1")
    plt.tight_layout()
    plt.show()

