"""Small, tested helpers for the exploration section of the notebook (analysis.py, section 4).

They describe the daily series; none of them is a forecaster. Inputs are plain pandas Series of
minutes under alert per day with a daily index.

* change_points      dates where the level of the series shifts (binary segmentation on mean shifts)
* acf                autocorrelation, optionally after removing a slowly moving level
* heavy_lift         how much more likely a heavy day is k days after a heavy day than on an average day
* dispersion_blocks  mean and standard deviation per block of days, and the slope of log std on log mean
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def change_points(x, min_size: int = 45, strength: float = 10.0) -> list[int]:
    """Positions i (first element of the right-hand segment) where the mean shifts.

    Binary segmentation: split a segment where the drop in squared error is largest, keep the split when
    the drop exceeds `strength * sigma2 * ln(n)`, then recurse on both sides. sigma2 is the noise variance
    estimated from day-to-day differences, so level shifts do not inflate it. Daily alert minutes are
    autocorrelated, so a plain BIC penalty (strength 2) would find too many breaks; 10 is conservative.
    Treat the result as a description of the data, not as a statistical test.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 2 * min_size:
        return []
    sigma2 = np.var(np.diff(x)) / 2.0
    pen = strength * sigma2 * np.log(n)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def sse(a, b):                                   # squared error around the mean of x[a:b]
        m = b - a
        s = c1[b] - c1[a]
        return (c2[b] - c2[a]) - s * s / m

    found: list[int] = []

    def split(a, b):
        if b - a < 2 * min_size:
            return
        ks = np.arange(a + min_size, b - min_size + 1)
        gains = np.array([sse(a, b) - sse(a, k) - sse(k, b) for k in ks])
        j = int(np.argmax(gains))
        if gains[j] > pen:
            k = int(ks[j])
            found.append(k)
            split(a, k)
            split(k, b)

    split(0, n)
    return sorted(found)


def acf(x: pd.Series, nlags: int = 60, detrend_window: int | None = None) -> pd.Series:
    """Autocorrelation at lags 1..nlags. With `detrend_window`, a centred rolling mean of that many days is
    subtracted first, so slow level changes do not show up as persistence."""
    x = pd.Series(x, dtype=float)
    if detrend_window:
        x = x - x.rolling(detrend_window, center=True, min_periods=detrend_window // 2).mean()
    return pd.Series([x.autocorr(k) for k in range(1, nlags + 1)], index=range(1, nlags + 1))


def heavy_lift(x: pd.Series, q: float = 0.9, lags=range(1, 8), local_window: int | None = None) -> pd.DataFrame:
    """P(heavy day at t+k | heavy day at t) against the overall share of heavy days.

    A day is heavy when it is above the q-quantile of the whole series, or, with `local_window`, above the
    q-quantile of the surrounding `local_window` days (centred), which takes out slow level changes.
    Columns: lag, p_heavy (overall share), p_given_heavy, lift (= p_given_heavy / p_heavy; 1 means no clustering).
    """
    x = pd.Series(x, dtype=float)
    if local_window:
        thr = x.rolling(local_window, center=True, min_periods=local_window // 2).quantile(q)
        valid = thr.notna()
        heavy = (x > thr) & valid
    else:
        heavy = x > x.quantile(q)
        valid = pd.Series(True, index=x.index)
    p = float(heavy[valid].mean())
    rows = []
    for k in lags:
        now, later = heavy, heavy.shift(-k)
        ok = valid & valid.shift(-k, fill_value=False) & now
        pg = float(later[ok].mean()) if ok.sum() else np.nan
        rows.append({"lag": k, "p_heavy": p, "p_given_heavy": pg, "lift": pg / p if p else np.nan})
    return pd.DataFrame(rows)


def dispersion_blocks(x: pd.Series, block: int = 30) -> tuple[pd.DataFrame, float]:
    """Mean and standard deviation of non-overlapping blocks of `block` days, and the slope of
    log(std) on log(mean) over the blocks with a positive mean and std.

    Slope near 0: spread does not depend on level. Near 1: spread grows in proportion to the level
    (multiplicative noise, so work on a log or square-root scale). Between: in between (Tweedie-like).
    """
    x = pd.Series(x, dtype=float)
    nb = len(x) // block
    g = np.arange(len(x)) // block
    d = x.iloc[: nb * block].groupby(g[: nb * block]).agg(["mean", "std"])
    d["start"] = [x.index[i * block] for i in range(nb)]
    d = d.set_index("start")
    ok = (d["mean"] > 0) & (d["std"] > 0)
    slope = float(np.polyfit(np.log(d.loc[ok, "mean"]), np.log(d.loc[ok, "std"]), 1)[0]) if ok.sum() >= 3 else np.nan
    return d, slope
