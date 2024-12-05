import typing as tp

import pytest
import numpy as np
from scipy.stats import chi, expon, norm


np.random.seed(1234)

from kupier_test import (
    kuiper_statistic_two_sided,
    kuiper_test,
    _kuiper_cdf
)

@pytest.mark.parametrize(
        "identical_samples",
    [
        ([1, 2, 3, 5], [5, 1, 2, 3]),
        (np.arange(10), np.arange(10))
    ]
)
def test_identical(identical_samples: tuple[tp.Iterable]):
    samples_1, samples_2 = identical_samples
    assert kuiper_statistic_two_sided(samples_1, samples_2) == 0

@pytest.mark.parametrize(
        "n",
        list(range(1000, 1020))
)
def test_kuiper_cdf(n):
    cdf_vals = np.array(
        [_kuiper_cdf(i, n) for i in np.arange(1, 5, 0.1)]
    )
    assert np.all((cdf_vals[1:] - cdf_vals[:-1]) > -1e-08)


@pytest.mark.parametrize(
    "samples, cdf",
    [
        (chi.rvs(2, size=100), chi(2).cdf),
        (chi.rvs(3, size=100), chi(3).cdf),
        (expon.rvs(1, 2, size=100), expon(1, 2).cdf),
        (norm.rvs(1, 2, size=100), norm(1, 2).cdf),
        (norm.rvs(3, 4, size=1000), norm(3, 4).cdf),
    ]
)
def test_onesided_big_pvalue(samples, cdf):
    assert kuiper_test(samples, cdf).p_value > 0.1


@pytest.mark.parametrize(
    "samples, cdf",
    [
        (chi.rvs(2, size=100), chi.rvs(2, size=100)),
        (chi.rvs(3, size=100), chi.rvs(3, size=100)),
        (expon.rvs(1, 2, size=100), expon.rvs(1, 2, size=100)),
        (norm.rvs(1, 2, size=100), norm.rvs(1, 2, size=100)),
        (norm.rvs(3, 4, size=1000), norm.rvs(3, 4, size=1000)),
    ]
)
def test_twosided_big_pvalue(samples, cdf):
    assert kuiper_test(samples, cdf).p_value > 0.1


@pytest.mark.parametrize(
    "samples, cdf",
    [
        (chi.rvs(2, size=100), chi(3).cdf),
        (chi.rvs(4, size=100), chi(3).cdf),
        (expon.rvs(1, 2, size=100), expon(1, 5).cdf),
        (norm.rvs(1, 2, size=100), norm(1, 5).cdf),
        (norm.rvs(3, 4, size=1000), norm(1, 5).cdf),
    ]
)
def test_onesided_small_pvalue(samples, cdf):
    assert kuiper_test(samples, cdf).p_value < 0.01


@pytest.mark.parametrize(
    "samples, cdf",
    [
        (chi.rvs(2, size=100), chi.rvs(3, size=100)),
        (chi.rvs(3, size=100), chi.rvs(5, size=100)),
        (expon.rvs(1, 2, size=100), expon.rvs(2, 4, size=100)),
        (norm.rvs(0, 5, size=100), norm.rvs(1, 2, size=100)),
        (norm.rvs(2, 4, size=1000), norm.rvs(1, 4, size=1000)),
    ]
)
def test_twosided_big_pvalue(samples, cdf):
    assert kuiper_test(samples, cdf).p_value < 0.01
