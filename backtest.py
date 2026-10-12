"""Rolling-origin backtest for the daily minutes-under-alert series, with simple recent-level forecasters.

Forecast origin = a local midnight `o`. A forecaster sees only days up to `o - 1` and predicts the days
`o .. o + horizon - 1` (horizon 1 .. 7). `run_backtest` truncates the series itself, so no method can see
the future; tests/test_backtest.py checks this by changing the future values and comparing.

Methods (name -> spec), all flat across the horizon unless noted:
    last              yesterday's value
    same_weekday      value 7 days before the target day (14 for horizon 8..14, ...)
    same_weekday_mean4 mean of the same weekday in the last 4 weeks
    mean_W, median_W  mean / median of the last W days (W = all: the whole history). median_W also gives
                      prediction intervals: empirical quantiles of the same W days
    ewma_H            exponentially weighted mean over the whole history, half-life H days
    since_break_mean, since_break_median
                      mean / median of the days since the last level shift detected with the data
                      available at the origin (explore.change_points; needs 45 days after a shift to see it)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import explore as ex

QS = (0.025, 0.1, 0.5, 0.9, 0.975)
WINDOWS = (7, 14, 30, 60, 90, 180, 365, 730, None)
HALF_LIVES = (3, 7, 14, 30, 60, 120)


def default_methods(windows=WINDOWS, half_lives=HALF_LIVES) -> dict:
    m = {"last": ("last",), "same_weekday": ("seasonal",), "same_weekday_mean4": ("wd4",)}
    for w in windows:
        tag = "all" if w is None else str(w)
        m[f"mean_{tag}"] = ("mean", w)
        m[f"median_{tag}"] = ("median", w)
    for h in half_lives:
        m[f"ewma_{h}"] = ("ewma", h)
    m["since_break_mean"] = ("since_break", "mean")
    m["since_break_median"] = ("since_break", "median")
    return m


def _last_break_start(hist: pd.Series, cache: dict) -> int:
    if "break" not in cache:
        cps = ex.change_points(np.sqrt(hist.to_numpy(float)), min_size=45, strength=10)
        cache["break"] = cps[-1] if cps else 0
    return cache["break"]


def forecast(hist: pd.Series, origin, horizon: int, spec: tuple, cache: dict | None = None):
    """Point forecasts (length horizon) and, for median methods, quantile forecasts (horizon x len(QS)) or None.

    `hist` must end at origin - 1 day (run_backtest guarantees it)."""
    cache = {} if cache is None else cache
    kind, args = spec[0], spec[1:]
    v = hist.to_numpy(float)
    targets = pd.date_range(pd.Timestamp(origin), periods=horizon, freq="D")
    q = None
    if kind == "last":
        point = np.full(horizon, v[-1])
    elif kind in ("seasonal", "wd4"):
        weeks = np.ceil((targets - (hist.index[-1])).days.to_numpy() / 7).astype(int)     # whole weeks back to land in the past
        point = np.empty(horizon)
        for i, (t, k) in enumerate(zip(targets, weeks)):
            lags = [k] if kind == "seasonal" else [k, k + 1, k + 2, k + 3]
            point[i] = hist.reindex([t - pd.Timedelta(days=7 * j) for j in lags]).mean()
    elif kind in ("mean", "median"):
        w = args[0]
        x = v if w is None else v[-w:]
        if kind == "mean":
            point = np.full(horizon, x.mean())
        else:
            qs = np.quantile(x, QS)
            point = np.full(horizon, qs[QS.index(0.5)])
            q = np.tile(qs, (horizon, 1))
    elif kind == "ewma":
        age = np.arange(len(v))[::-1]
        wts = 0.5 ** (age / args[0])
        point = np.full(horizon, np.average(v, weights=wts))
    elif kind == "since_break":
        x = v[_last_break_start(hist, cache):]
        point = np.full(horizon, x.mean() if args[0] == "mean" else np.median(x))
    else:
        raise ValueError(f"unknown method {spec}")
    return point, q


def run_backtest(y: pd.Series, origins, horizon: int = 7, methods: dict | None = None) -> pd.DataFrame:
    """Long table: origin, h, target, method, y, point, q025, q10, q50, q90, q975 (quantiles only for median methods).

    Origins whose targets are not all in `y` are skipped."""
    methods = methods or default_methods()
    y = y.astype(float)
    cols = {k: [] for k in ("origin", "h", "target", "method", "y", "point", *[f"q{q}" for q in QS])}
    for o in pd.DatetimeIndex(origins):
        targets = pd.date_range(o, periods=horizon, freq="D")
        if targets[-1] > y.index[-1] or o - pd.Timedelta(days=1) < y.index[0]:
            continue
        hist = y.loc[: o - pd.Timedelta(days=1)]               # the only thing a forecaster may see
        actual = y.reindex(targets).to_numpy()
        cache: dict = {}
        for name, spec in methods.items():
            point, q = forecast(hist, o, horizon, spec, cache)
            cols["origin"].append(np.full(horizon, o.to_datetime64()))
            cols["h"].append(np.arange(1, horizon + 1))
            cols["target"].append(targets.to_numpy())
            cols["method"].append(np.full(horizon, name, dtype=object))
            cols["y"].append(actual)
            cols["point"].append(point)
            for j, qq in enumerate(QS):
                cols[f"q{qq}"].append(q[:, j] if q is not None else np.full(horizon, np.nan))
    return pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()}) if cols["y"] else pd.DataFrame(columns=list(cols))


def naive_scale(y: pd.Series) -> float:
    """Mean absolute day-to-day change; the unit for MASE (computed once, on data before the test period)."""
    return float(np.abs(np.diff(y.to_numpy(float))).mean())


def score(bt: pd.DataFrame, scale: float, by=("method",)) -> pd.DataFrame:
    """MAE, MASE (= MAE / scale), RMSE, bias (forecast minus actual) per group."""
    d = bt.assign(err=bt["point"] - bt["y"])
    d["ae"], d["se"] = d["err"].abs(), d["err"] ** 2
    g = d.groupby(list(by))
    out = pd.DataFrame({"mae": g["ae"].mean(), "rmse": np.sqrt(g["se"].mean()), "bias": g["err"].mean(), "n": g.size()})
    out["mase"] = out["mae"] / scale
    return out[["mae", "mase", "rmse", "bias", "n"]]


def interval_score(bt: pd.DataFrame, by=("method",)) -> pd.DataFrame:
    """Coverage and mean interval (Winkler) score of the 80% (q10..q90) and 95% (q025..q975) intervals; median methods only."""
    d = bt.dropna(subset=["q0.1"]).copy()
    out = {}
    for lvl, lo, hi, alpha in ((80, "q0.1", "q0.9", 0.2), (95, "q0.025", "q0.975", 0.05)):
        width = d[hi] - d[lo]
        miss = (d[lo] - d["y"]).clip(lower=0) + (d["y"] - d[hi]).clip(lower=0)
        d[f"cov{lvl}"] = ((d["y"] >= d[lo]) & (d["y"] <= d[hi])).astype(float)
        d[f"is{lvl}"] = width + (2 / alpha) * miss
        d[f"width{lvl}"] = width
    g = d.groupby(list(by))
    return g[["cov80", "cov95", "width80", "width95", "is80", "is95"]].mean().join(g.size().rename("n"))


def dm_test(loss_a, loss_b, lags: int = 6) -> tuple[float, float, float]:
    """Diebold-Mariano test on per-origin losses of two methods (one number per origin, aligned).

    Returns (mean loss a minus loss b, statistic, two-sided p-value). A positive mean means a is worse.
    The variance uses Newey-West weights with `lags` lags because origins one day apart share days of a
    7-day horizon (use lags = horizon - 1). Normal approximation; no correction for running many tests.
    """
    d = (np.asarray(loss_a, float) - np.asarray(loss_b, float))
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mu = d.mean()
    dc = d - mu
    var = (dc * dc).mean()
    for k in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - k / (lags + 1)) * (dc[k:] * dc[:-k]).mean()
    se = math.sqrt(max(var, 1e-12) / n)
    stat = mu / se
    p = math.erfc(abs(stat) / math.sqrt(2))
    return float(mu), float(stat), float(p)
