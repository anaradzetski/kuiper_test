import typing as tp
import numpy as np
from dataclasses import dataclass
from itertools import count
from collections.abc import Iterable


def kuiper_statistic_one_sided(
    samples: tp.Iterable[float],
    cdf: tp.Callable[[np.ndarray[float]], np.ndarray[float]]
) -> float:
    sorted_samples = np.sort(np.array(samples))
    n = len(sorted_samples)
    cdf_vals = cdf(sorted_samples)
    if np.max(cdf_vals) > 1 or np.min(cdf_vals) < 0 or np.any(cdf_vals[:-1] > cdf_vals[1:]):
        return ValueError("Bad cdf function")
    D_plus = np.max(np.arange(1, n + 1) / n - cdf_vals)
    D_minus = np.max(cdf_vals - np.arange(n) / n)
    return np.sqrt(n) * (D_plus + D_minus)


def kuiper_statistic_two_sided(
    samples_1: tp.Iterable[float],
    samples_2: tp.Iterable[float]
) -> float:
    numbered_samples_1, numbered_samples_2 = [(i, 1) for i in samples_1], [(i, 2) for i in samples_2]
    joint_samples = sorted(
        numbered_samples_1 + numbered_samples_2,
        key=lambda x: x[0]
    )
    n_1, n_2 = len(numbered_samples_1), len(numbered_samples_2)
    D_plus, D_minus = -np.inf, -np.inf
    cur_diff, max_num = 0, -np.inf
    for val, idx in joint_samples:
        if val > max_num:
            D_plus = max(cur_diff, D_plus)
            D_minus = max(-cur_diff, D_minus)
        cur_diff += 1 / n_1 if idx == 1 else -1 / n_2
        max_num = val
    n, m = len(samples_1), len(samples_2)
    return np.sqrt(n * m / (n + m)) * (D_plus + D_minus)


def _kuiper_cdf(x: float, n: int, precision: float = 1e-20) -> float:
    assert precision > 0
    if x <= 0:
        return 0
    cur_summand = np.inf
    first_sum = 0
    for k in count(1):
        if np.abs(cur_summand) < precision:
            break
        cur_summand = 2 * (-1) ** (k - 1) * (4 * k ** 2 * x ** 2 - 1) * np.exp(-2 * k ** 2 * x ** 2)
        first_sum += cur_summand
    second_sum = 0
    cur_summand = np.inf
    coeff = 8 / (3 * np.sqrt(n)) * x
    for k in count(1):
        if np.abs(cur_summand) < precision:
            break
        cur_summand = coeff * k ** 2 * (4 * k ** 2 * x ** 2 - 3) * np.exp(-2 * k ** 2 * x ** 2)
        second_sum += cur_summand
    return 1 - first_sum + second_sum

@dataclass
class KuiperTestResult:
    statistic: float
    p_value: float


def kuiper_test(samples: tp.Iterable, cdf: tp.Iterable[str] | tp.Callable[[np.ndarray[float]], np.ndarray[float]]):
    if isinstance(cdf, Iterable):
        kuiper_stat = kuiper_statistic_two_sided(samples, cdf)
        n = len(samples) * len(cdf) / (len(samples) + len(cdf))
    else:
        kuiper_stat = kuiper_statistic_one_sided(samples, cdf)
        n = len(samples)
    p_value = 1 - _kuiper_cdf(kuiper_stat, n)
    return KuiperTestResult(statistic=kuiper_stat, p_value=p_value)

