import numpy as np
from scipy.interpolate import interp1d


def maybe_interpolate_line(ts_raw, ts_line, d_line, *, extrapolate=False):
    """
    Returns d_line resampled to ts_raw.
    - Drops NaNs and duplicate timestamps from ts_line/d_line
    - Sorts by time
    - Clamps outside range by default; set extrapolate=True for linear extrapolation
    """
    ts_raw = np.asarray(ts_raw)
    ts_line = np.asarray(ts_line)
    d_line = np.asarray(d_line)

    if ts_raw.ndim != 1 or ts_line.ndim != 1 or d_line.ndim != 1:
        raise ValueError("All inputs must be 1D arrays")

    if ts_line.size != d_line.size:
        raise ValueError("ts_line and d_line must have the same length")

    if ts_raw.size == 0:
        return np.array([], dtype=d_line.dtype)

    # Fast path only if lengths match and timestamps align exactly
    if ts_raw.size == ts_line.size and np.array_equal(ts_raw, ts_line):
        return d_line.copy()

    # Mask out NaNs
    valid = ~(np.isnan(ts_line) | np.isnan(d_line))
    ts_line_valid = ts_line[valid]
    d_line_valid = d_line[valid]

    if ts_line_valid.size == 0:
        # Nothing to interpolate from; return NaNs matching ts_raw length
        return np.full(ts_raw.shape, np.nan, dtype=float)

    # Sort and deduplicate ts_line
    order = np.argsort(ts_line_valid)
    ts_sorted = ts_line_valid[order]
    d_sorted = d_line_valid[order]

    # Keep first occurrence for duplicates
    unique_ts, unique_idx = np.unique(ts_sorted, return_index=True)
    d_unique = d_sorted[unique_idx]

    if unique_ts.size == 1:
        # Only one point: constant line
        return np.full(ts_raw.shape, d_unique[0], dtype=d_unique.dtype)

    if extrapolate:
        f = interp1d(unique_ts, d_unique, kind="linear", bounds_error=False, fill_value="extrapolate", assume_sorted=True)
    else:
        # Clamp to edge values
        f = interp1d(unique_ts, d_unique, kind="linear", bounds_error=False, fill_value=(d_unique[0], d_unique[-1]), assume_sorted=True)

    d_line_interp = f(ts_raw)

    if d_line_interp.shape[0] != ts_raw.shape[0]:
        raise RuntimeError("Interpolation produced unexpected shape")

    return d_line_interp

# Interpolate top lines to match number of pings
"""
d_top_true = np.interp(
    np.linspace(0, len(d_top_true), num_pings),
    np.arange(len(d_top_true)),
    d_top_true,
)
d_top_gen = np.interp(
    np.linspace(0, len(d_top_gen), num_pings),
    np.arange(len(d_top_gen)),
    d_top_gen,
)
"""    

"""
d_bottom_true = np.interp(
    np.linspace(0, len(d_bottom_true_raw), num_pings),
    np.arange(len(d_bottom_true_raw)),
    d_bottom_true_raw,
)
"""