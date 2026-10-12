"""Forecasting models for daily minutes under alert, run in the same rolling-origin setup as backtest.py.

Every method is named <family>_<inputs>_<window>, for example ridge_full_365 or ets_sqrt_all:

    ets_sqrt_W     exponential smoothing of the level, ETS(A,N,N), on the square root of minutes over the last W days.
                   The smoothing weight is picked by the smallest one-step squared error; the start level is the first value.
    arima_sqrt_W   ARIMA(1,1,1) on the square root over the last W days (statsmodels); parameters refitted weekly.
    ridge_level_W  ridge regression on the square root of minutes; inputs: recent-level features, weekday, horizon.
    ridge_full_W   the same plus weather on the target day and the holiday features.
    hgb_level_W    gradient-boosted trees (scikit-learn HistGradientBoostingRegressor) with a Poisson loss, which suits
                   skewed, zero-heavy minutes; inputs as ridge_level.
    hgb_full_W     the same plus weather and holidays.
                   (scikit-learn ships its own OpenMP library on macOS, so `uv sync` is enough; LightGBM was dropped
                   because its macOS build needs a separate system library, libomp.)
W is a number of days or "all". Feature models are trained on (origin, horizon) rows whose target day is before the
refit origin, refitted every `refit_every` days; between refits the frozen model is applied to fresh features.

What a forecaster may see: the recent-level features of origin o use only days up to o - 1 (they are rolling
statistics shifted by one day). Weather on the target day comes from the archived Open-Meteo forecasts, which
are close to same-day forecasts, so backtest gains from weather at horizons 2 to 7 are optimistic (README).

Intervals for every method (baselines included) come from `add_intervals`: the method's own past out-of-sample
errors in the last `lookback` days, measured on the scale minutes ** POWER, added to the point forecast on that
scale and transformed back. With POWER below 1 the bands widen when the level is high (level-scaled intervals).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import backtest as bk

DAY = pd.Timedelta(days=1)
LEVEL_COLS = ["last1", "mean7", "mean14", "mean30", "mean90", "med7", "med30", "ewm7", "ewm30", "std30", "max7", "zero30"]
WEATHER_COLS = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max", "cloud_cover_mean", "daylight_min"]
HOLIDAY_GROUPS = ["ua_statehood", "ua_military", "ru_state_military", "religious", "new_year", "war_anniversary"]
HOLIDAY_COLS = [f"{g}_{s}" for g in HOLIDAY_GROUPS for s in ("is_day", "eve", "within2")]
POWER = 0.3            # interval scale: 1 - 0.70 (spread grows like level ** 0.70, so level ** 0.3 has steady spread)

MODEL_GRID = (
    [f"ets_sqrt_{w}" for w in ("30", "90", "365", "all")]
    + [f"arima_sqrt_{w}" for w in ("90", "365", "all")]
    + [f"ridge_{f}_{w}" for f in ("level", "full") for w in ("90", "180", "365", "all")]
    + [f"hgb_{f}_{w}" for f in ("level", "full") for w in ("180", "365", "all")]
)


def parse(name: str) -> tuple[str, str, int | None]:
    fam, inputs, w = name.split("_")
    return fam, inputs, (None if w == "all" else int(w))


# ---------------------------------------------------------------- features
def level_features(y: pd.Series) -> pd.DataFrame:
    """Recent-level features, indexed by forecast origin (the day after the last day they use)."""
    s = y.astype(float)
    f = pd.DataFrame({
        "last1": s,
        "mean7": s.rolling(7).mean(), "mean14": s.rolling(14).mean(), "mean30": s.rolling(30).mean(),
        "mean90": s.rolling(90).mean(),
        "med7": s.rolling(7).median(), "med30": s.rolling(30).median(),
        "ewm7": s.ewm(halflife=7).mean(), "ewm30": s.ewm(halflife=30).mean(),
        "std30": s.rolling(30).std(), "max7": s.rolling(7).max(), "zero30": (s == 0).astype(float).rolling(30).mean(),
    })
    f.index = f.index + DAY
    f.index.name = "origin"
    return f


def make_rows(y: pd.Series, exo: pd.DataFrame | None, origins, horizon: int) -> pd.DataFrame:
    """One row per (origin, horizon): level features at the origin, calendar and exogenous inputs of the target day,
    and the true value of the target day (NaN when unknown)."""
    lf = level_features(y)
    origins = pd.DatetimeIndex(origins)
    parts = []
    for h in range(1, horizon + 1):
        target = origins + (h - 1) * DAY
        r = lf.reindex(origins).reset_index(drop=True)
        r.insert(0, "origin", origins)
        r.insert(1, "h", h)
        r.insert(2, "target", target)
        r["weekday"] = target.dayofweek
        r["y"] = y.reindex(target).to_numpy(float)
        if exo is not None:
            cols = [c for c in WEATHER_COLS + HOLIDAY_COLS if c in exo.columns]
            r[cols] = exo.reindex(target)[cols].to_numpy(float)
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


def feature_cols(inputs: str, rows: pd.DataFrame) -> list[str]:
    cols = LEVEL_COLS + ["h", "weekday"]
    if inputs == "full":
        cols += [c for c in WEATHER_COLS + HOLIDAY_COLS if c in rows.columns]
    return cols


# ---------------------------------------------------------------- feature models
class RidgeSqrt:
    """Ridge regression on sqrt(minutes); level features enter as square roots, weekday as dummies."""

    def __init__(self, cols, alpha: float = 3.0):
        self.cols, self.alpha = cols, alpha

    def _design(self, r):
        X = r[[c for c in self.cols if c != "weekday"]].astype(float).copy()
        for c in LEVEL_COLS:
            if c in X and c != "zero30":
                X[c] = np.sqrt(X[c].clip(lower=0))
        if "weekday" in self.cols:
            for d in range(1, 7):
                X[f"wd{d}"] = (r["weekday"] == d).astype(float)
        return X

    def fit(self, r):
        X = self._design(r)
        self.fill = X.mean()
        X = X.fillna(self.fill)
        self.mu, self.sd = X.mean(), X.std().replace(0, 1).fillna(1)
        Z = ((X - self.mu) / self.sd).to_numpy()
        t = np.sqrt(r["y"].to_numpy(float))
        self.b0 = t.mean()
        self.b = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ (t - self.b0))
        return self

    def predict(self, r):
        Z = ((self._design(r).fillna(self.fill) - self.mu) / self.sd).to_numpy()
        return np.clip(self.b0 + Z @ self.b, 0, None) ** 2


class HGBPoisson:
    """Gradient-boosted trees with a Poisson loss (predicts the mean minutes)."""

    def __init__(self, cols, seed: int = 0):
        self.cols, self.seed = cols, seed

    def fit(self, r):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.m = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=150, max_leaf_nodes=15,
                                               min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
                                               random_state=self.seed)
        self.m.fit(r[self.cols].astype(float).to_numpy(), r["y"].to_numpy(float))
        return self

    def predict(self, r):
        return np.clip(self.m.predict(r[self.cols].astype(float).to_numpy()), 0, None)


def _make_feature_model(fam, inputs, rows):
    cols = feature_cols(inputs, rows)
    return RidgeSqrt(cols) if fam == "ridge" else HGBPoisson(cols)


def _train_rows(rows, refit_origin, window):
    t = rows[(rows["target"] < refit_origin) & rows["y"].notna() & rows["mean90"].notna()]
    if window is not None:
        t = t[t["target"] >= refit_origin - window * DAY]
    return t


# ---------------------------------------------------------------- series models
def ses_sqrt(x: np.ndarray, alphas=np.linspace(0.02, 0.98, 49)) -> float:
    """ETS(A,N,N) on sqrt values: smoothing weight with the smallest one-step squared error; returns the last level."""
    s = np.sqrt(np.clip(x, 0, None))
    lev = np.full(len(alphas), s[0])
    sse = np.zeros(len(alphas))
    for v in s[1:]:
        e = v - lev
        sse += e * e
        lev = lev + alphas * e
    return float(lev[int(np.argmin(sse))])


def _arima_fit(x):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(np.sqrt(np.clip(x, 0, None)), order=(1, 1, 1)).fit(disp=False)


def _arima_apply(res, x, horizon):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = res.apply(np.sqrt(np.clip(x, 0, None)))
        return np.clip(r.forecast(horizon), 0, None) ** 2


# ---------------------------------------------------------------- backtest
def run_models(y: pd.Series, exo: pd.DataFrame | None, origins, horizon: int = 7, names=MODEL_GRID,
               refit_every: int = 7, progress=None) -> pd.DataFrame:
    """Long table like backtest.run_backtest (origin, h, target, method, y, point); no intervals yet."""
    y = y.astype(float)
    origins = pd.DatetimeIndex(sorted(origins))
    origins = origins[(origins + (horizon - 1) * DAY <= y.index[-1]) & (origins - DAY >= y.index[0])]
    need_rows = any(parse(n)[0] in ("ridge", "hgb") for n in names)
    rows = make_rows(y, exo, pd.date_range(y.index[0] + DAY, origins[-1]), horizon) if need_rows else None
    out = []
    for k, name in enumerate(names):
        fam, inputs, window = parse(name)
        if progress:
            progress(f"{k + 1}/{len(names)} {name}")
        recs = []
        for i in range(0, len(origins), refit_every):
            block = origins[i:i + refit_every]
            R = block[0]
            if fam in ("ridge", "hgb"):
                model = _make_feature_model(fam, inputs, rows).fit(_train_rows(rows, R, window))
                p = rows[rows["origin"].isin(block)]
                recs.append(p[["origin", "h", "target", "y"]].assign(point=model.predict(p)))
                continue
            res = None
            for o in block:
                hist = y.loc[: o - DAY].to_numpy()
                x = hist if window is None else hist[-window:]
                if fam == "ets":
                    point = np.full(horizon, ses_sqrt(x) ** 2)
                else:
                    if res is None:
                        res = _arima_fit(y.loc[: R - DAY].to_numpy()[-window:] if window else y.loc[: R - DAY].to_numpy())
                    point = _arima_apply(res, x, horizon)
                tg = pd.date_range(o, periods=horizon, freq="D")
                recs.append(pd.DataFrame({"origin": o, "h": np.arange(1, horizon + 1), "target": tg,
                                          "y": y.reindex(tg).to_numpy(), "point": point}))
        out.append(pd.concat(recs, ignore_index=True).assign(method=name))
    df = pd.concat(out, ignore_index=True)
    return df[["origin", "h", "target", "method", "y", "point"]]


def forecast_final(name: str, y: pd.Series, exo: pd.DataFrame | None, origin, horizon: int) -> np.ndarray:
    """Point forecasts for origin .. origin + horizon - 1 from a model fitted on all of y (which must end at origin - 1)."""
    y = y.astype(float)
    origin = pd.Timestamp(origin)
    assert y.index[-1] == origin - DAY, "y must end the day before the origin"
    fam, inputs, window = parse(name)
    hist = y.to_numpy()
    x = hist if window is None else hist[-window:]
    if fam == "ets":
        return np.full(horizon, ses_sqrt(x) ** 2)
    if fam == "arima":
        return _arima_apply(_arima_fit(x), x, horizon)
    rows = make_rows(y, exo, pd.date_range(y.index[0] + DAY, origin), horizon)
    model = _make_feature_model(fam, inputs, rows).fit(_train_rows(rows, origin, window))
    p = rows[rows["origin"] == origin].sort_values("h")
    return model.predict(p)


# ---------------------------------------------------------------- intervals
DAY_MINUTES = 1440.0   # a day cannot hold more minutes than this (1500 only on the October clock-change day)


def band(point, resid, power: float = POWER, cap: float = DAY_MINUTES) -> np.ndarray:
    """Quantile forecasts (n x len(QS)) from point forecasts and past errors measured on the scale minutes ** power,
    kept between 0 and `cap` minutes."""
    q = np.quantile(resid, bk.QS)
    z = np.clip(np.asarray(point, float), 0, None)[:, None] ** power + q[None, :]
    return np.clip(np.clip(z, 0, None) ** (1 / power), 0, cap)


def _errors(known, power):
    return np.clip(known["y"].to_numpy(), 0, None) ** power - np.clip(known["point"].to_numpy(), 0, None) ** power


def add_intervals(df: pd.DataFrame, power: float = POWER, lookback: int = 90, min_n: int = 28,
                  mode: str = "recent", ratio: float = 2.0) -> pd.DataFrame:
    """Fill q0.025 .. q0.975 for every method from its own past errors (targets already over at the origin).

    mode "recent": errors on targets in [origin - lookback, origin - 1].
    mode "level":  regime-aware; errors from the same look-back whose point forecast was within a factor `ratio`
                   of this origin's point forecast (so a high-level forecast borrows errors from high-level days).
                   Falls back to the last 90 days when fewer than `min_n` such errors exist.
    Rows with fewer than `min_n` usable errors get no interval (NaN)."""
    df = df.sort_values(["method", "origin", "h"]).reset_index(drop=True)
    qcols = [f"q{q}" for q in bk.QS]
    for c in qcols:
        df[c] = np.nan
    for m, g in df.groupby("method", sort=False):
        known = g.dropna(subset=["y"])
        tgt = known["target"].to_numpy()
        pts = np.clip(known["point"].to_numpy(), 0, None)
        res = _errors(known, power)
        for o, idx in g.groupby("origin").groups.items():
            o64 = np.datetime64(o)
            past = tgt < o64
            sel = past & (tgt >= o64 - np.timedelta64(lookback, "D"))
            p_now = df.loc[idx, "point"].to_numpy()
            if mode == "level":
                lvl = max(float(np.mean(p_now)), 1.0)
                near = sel & (pts >= lvl / ratio) & (pts <= lvl * ratio)
                sel = near if near.sum() >= min_n else past & (tgt >= o64 - np.timedelta64(90, "D"))
            if sel.sum() < min_n:
                continue
            df.loc[idx, qcols] = band(p_now, res[sel], power)
    return df


def combine(df: pd.DataFrame, a: str, b: str, name: str, weights: str = "equal", lookback: int = 30,
            min_n: int = 14) -> pd.DataFrame:
    """Forecast combination w * a + (1 - w) * b as new rows named `name` (point only; add intervals afterwards).

    weights "equal": w = 0.5.
    weights "adaptive": w = (1 / MAE_a) / (1 / MAE_a + 1 / MAE_b), MAE over targets in [origin - lookback, origin - 1]
                        that are already over at the origin; w = 0.5 when fewer than `min_n` such errors exist."""
    cols = ["origin", "h", "target", "y"]
    pa = df[df["method"] == a].set_index(["origin", "h"])
    pb = df[df["method"] == b].set_index(["origin", "h"])
    j = pa[["target", "y", "point"]].join(pb[["point"]], rsuffix="_b", how="inner").reset_index()
    w = pd.Series(0.5, index=j.index)
    if weights == "adaptive":
        ea = pa.dropna(subset=["y"]).assign(ae=lambda d: (d["point"] - d["y"]).abs())[["target", "ae"]]
        eb = pb.dropna(subset=["y"]).assign(ae=lambda d: (d["point"] - d["y"]).abs())[["target", "ae"]]
        ta, aa = ea["target"].to_numpy(), ea["ae"].to_numpy()
        tb, ab = eb["target"].to_numpy(), eb["ae"].to_numpy()
        for o, idx in j.groupby("origin").groups.items():
            o64 = np.datetime64(o)
            lo = o64 - np.timedelta64(lookback, "D")
            sa, sb = (ta < o64) & (ta >= lo), (tb < o64) & (tb >= lo)
            if sa.sum() >= min_n and sb.sum() >= min_n:
                ia, ib = 1 / max(aa[sa].mean(), 1e-9), 1 / max(ab[sb].mean(), 1e-9)
                w.loc[idx] = ia / (ia + ib)
    out = j[cols].copy()
    out["method"] = name
    out["point"] = w.to_numpy() * j["point"].to_numpy() + (1 - w.to_numpy()) * j["point_b"].to_numpy()
    out["weight_a"] = w.to_numpy()
    return out[["origin", "h", "target", "method", "y", "point", "weight_a"]]


def recent_errors(df: pd.DataFrame, method: str, before, power: float = POWER, lookback: int = 90,
                  mode: str = "recent", ratio: float = 2.0, point_now: float | None = None, min_n: int = 28) -> np.ndarray:
    """The errors `add_intervals` would use for an origin `before` (for the final forecast's band)."""
    g = df[(df["method"] == method) & df["y"].notna()]
    before = pd.Timestamp(before)
    past = g[g["target"] < before]
    sel = past[past["target"] >= before - lookback * DAY]
    if mode == "level":
        lvl = max(float(point_now), 1.0)
        near = sel[(sel["point"] >= lvl / ratio) & (sel["point"] <= lvl * ratio)]
        sel = near if len(near) >= min_n else past[past["target"] >= before - 90 * DAY]
    return _errors(sel, power)


def combine_weight(df: pd.DataFrame, a: str, b: str, before, weights: str = "equal", lookback: int = 30,
                   min_n: int = 14) -> float:
    """The weight on `a` that `combine` would use at origin `before` (for the final forecast)."""
    if weights == "equal":
        return 0.5
    before = pd.Timestamp(before)
    maes = []
    for m in (a, b):
        g = df[(df["method"] == m) & df["y"].notna() & (df["target"] < before) & (df["target"] >= before - lookback * DAY)]
        if len(g) < min_n:
            return 0.5
        maes.append(max((g["point"] - g["y"]).abs().mean(), 1e-9))
    ia, ib = 1 / maes[0], 1 / maes[1]
    return ia / (ia + ib)
