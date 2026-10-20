"""Small statistics helpers shared by the load tool, the canary comparison and the gates."""

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]. Empty input raises (never invent a number)."""
    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
