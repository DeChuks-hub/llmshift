"""Small statistics helpers: confidence ranges for pass rates, paired tests, sample sizes."""

from __future__ import annotations

import math
import statistics

Z95 = 1.959964


def wilson_interval(successes: float, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (well behaved at 0% and 100%)."""
    if n <= 0:
        return 0.0, 1.0
    p = min(1.0, max(0.0, successes / n))
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def pass_rate_interval(per_case: list[float], z: float = Z95) -> tuple[float, float, float]:
    """(mean, low, high) for a pass rate measured over cases, each with a pass fraction
    across repeated runs. The cases are the sample, so the range reflects how many
    cases there are as well as how consistently each one passes."""
    n = len(per_case)
    if n == 0:
        return 0.0, 0.0, 1.0
    mean = sum(per_case) / n
    lo, hi = wilson_interval(mean * n, n, z)
    if n >= 2:
        se = statistics.stdev(per_case) / math.sqrt(n)
        lo, hi = min(lo, mean - z * se), max(hi, mean + z * se)  # the wider of the two
    return mean, max(0.0, lo), min(1.0, hi)


def margin_of_error(p: float, n: int, z: float = Z95) -> float:
    if n <= 0:
        return 1.0
    p = min(1.0, max(0.0, p))
    return z * math.sqrt(max(p * (1 - p), 0.25 / n) / n)


def cases_needed(p: float, margin: float = 0.05, z: float = Z95) -> int:
    """How many cases make the Wilson range of a pass rate `p` at most +/- `margin` wide.

    The usual p(1-p)/margin^2 rule says 16 cases near 100%, where 16 of 16 still only
    shows 81%-100%; the Wilson half-width shrinks steadily with n, so search it.
    """
    p = min(1.0, max(0.0, p))
    margin = max(margin, 1e-4)

    def half(n: int) -> float:
        return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)

    lo, hi = 1, 1
    while half(hi) > margin:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if half(mid) > margin:
            lo = mid + 1
        else:
            hi = mid
    return lo


def mcnemar_p(b: int, c: int) -> float:
    """One-sided exact McNemar p-value for b regressions vs c improvements."""
    n = b + c
    if n == 0 or b == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / (2**n)


def two_proportion_p(p1: float, n1: int, p2: float, n2: int) -> float:
    """One-sided p-value that pass rate p2 (of n2 cases) is lower than p1 (of n1)."""
    if not n1 or not n2:
        return 1.0
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0 if p1 > p2 else 1.0
    return 0.5 * math.erfc(((p1 - p2) / se) / math.sqrt(2))
