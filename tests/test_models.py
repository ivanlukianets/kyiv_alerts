import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import models as md  # noqa: E402

DAY = pd.Timedelta(days=1)


def series(n=400, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="D")
    level = 60 + 40 * np.sin(np.arange(n) / 40)
    return pd.Series(rng.gamma(2.0, level / 2).round(), index=idx)


def exo_for(y, extra=20):
    idx = pd.date_range(y.index[0], y.index[-1] + extra * DAY)
    rng = np.random.default_rng(1)
    e = pd.DataFrame({c: rng.normal(size=len(idx)) for c in md.WEATHER_COLS}, index=idx)
    for c in md.HOLIDAY_COLS:
        e[c] = (rng.random(len(idx)) < 0.05).astype(float)
    return e


def test_level_features_use_only_the_past():
    y = series()
    o = y.index[200]
    full = md.level_features(y).loc[o]
    cut = md.level_features(y.loc[: o - DAY]).loc[o]
    pd.testing.assert_series_equal(full, cut)
    assert full["last1"] == y.loc[o - DAY]
    assert np.isclose(full["mean7"], y.loc[o - 7 * DAY: o - DAY].mean())


def test_rows_align_targets_and_inputs():
    y = series(100)
    e = exo_for(y)
    r = md.make_rows(y, e, [y.index[50]], 7)
    assert list(r["h"]) == list(range(1, 8))
    assert (r["target"] == r["origin"] + (r["h"] - 1) * DAY).all()
    assert np.allclose(r["y"], y.iloc[50:57].to_numpy())
    assert np.allclose(r["temperature_2m_mean"], e.loc[r["target"], "temperature_2m_mean"].to_numpy())
    assert (r["mean7"] == r["mean7"].iloc[0]).all()                 # same origin, same level features


def test_ses_sqrt():
    assert np.isclose(md.ses_sqrt(np.full(50, 49.0)), 7.0)
    step = np.r_[np.full(60, 16.0), np.full(60, 100.0)]
    assert abs(md.ses_sqrt(step) - 10.0) < 0.1


def test_ridge_learns_a_level_relation():
    y = series(500)
    r = md.make_rows(y, None, pd.date_range(y.index[100], y.index[-8]), 1).dropna()
    m = md.RidgeSqrt(md.feature_cols("level", r)).fit(r)
    p = m.predict(r)
    assert np.corrcoef(p, r["mean30"])[0, 1] > 0.8 and (p >= 0).all()


def test_models_cannot_see_the_future():
    y = series(420)
    e = exo_for(y)
    origins = pd.date_range(y.index[300], y.index[400], freq="D")
    names = ["ets_sqrt_90", "arima_sqrt_90", "ridge_full_all", "hgb_level_all"]
    a = md.run_models(y, e, origins, 7, names)
    cut = origins[35]                                   # a refit origin (refit every 7 days from origins[0])
    y2 = y.copy()
    y2.loc[cut:] = y2.loc[cut:] * 3 + 500
    b = md.run_models(y2, e, origins, 7, names)
    ka, kb = a[a["origin"] <= cut], b[b["origin"] <= cut]
    assert len(ka) > 0
    assert np.allclose(ka["point"].to_numpy(), kb["point"].to_numpy())
    assert not np.allclose(a["point"], b["point"])


def test_final_forecast_matches_backtest_at_a_refit_origin():
    y = series(420)
    e = exo_for(y)
    origins = pd.date_range(y.index[300], y.index[400], freq="D")
    for name in ["ridge_full_365", "ets_sqrt_30", "hgb_full_all"]:
        bt = md.run_models(y, e, origins, 7, [name])
        o = origins[14]
        f = md.forecast_final(name, y.loc[: o - DAY], e, o, 7)
        assert np.allclose(bt.loc[bt["origin"] == o, "point"].to_numpy(), f)


def test_intervals_use_only_past_errors_and_cover():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2025-01-01", periods=700, freq="D")
    y = pd.Series(rng.gamma(2.0, 50, 700), index=idx)
    med = float(np.median(rng.gamma(2.0, 50, 100000)))
    rows = []
    for o in idx[30:-7]:
        tg = pd.date_range(o, periods=7)
        rows.append(pd.DataFrame({"origin": o, "h": range(1, 8), "target": tg, "method": "m", "y": y.reindex(tg).to_numpy(), "point": med}))
    df = md.add_intervals(pd.concat(rows, ignore_index=True))
    iv = df.dropna(subset=["q0.1"])
    cov80 = ((iv["y"] >= iv["q0.1"]) & (iv["y"] <= iv["q0.9"])).mean()
    assert abs(cov80 - 0.8) < 0.05
    o = iv["origin"].iloc[len(iv) // 2]
    df2 = df.copy()
    df2.loc[df2["target"] >= o, "y"] += 10000                # change the future
    b = md.add_intervals(df2[["origin", "h", "target", "method", "y", "point"]])
    a_ = df[df["origin"] == o]["q0.9"].to_numpy()
    b_ = b[b["origin"] == o]["q0.9"].to_numpy()
    assert np.allclose(a_, b_)


def _toy(n=300, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    y = pd.Series(rng.gamma(2.0, 50, n), index=idx)
    rows = []
    for o in idx[1:-7]:
        tg = pd.date_range(o, periods=7)
        for m, p in (("good", 100.0), ("bad", 300.0)):
            rows.append(pd.DataFrame({"origin": o, "h": range(1, 8), "target": tg, "method": m,
                                      "y": y.reindex(tg).to_numpy(), "point": p}))
    return pd.concat(rows, ignore_index=True)


def test_combine_equal_and_adaptive():
    df = _toy()
    eq = md.combine(df, "good", "bad", "c_eq")
    assert np.allclose(eq["point"], 200.0)
    ad = md.combine(df, "good", "bad", "c_ad", weights="adaptive", lookback=30)
    late = ad[ad["origin"] >= ad["origin"].min() + pd.Timedelta(days=40)]
    assert (late["weight_a"] > 0.5).all()                        # the more accurate method gets more weight
    early = ad[ad["origin"] == ad["origin"].min()]
    assert np.allclose(early["weight_a"], 0.5)                    # no past errors yet


def test_combine_uses_only_past_errors():
    df = _toy()
    o = df["origin"].iloc[len(df) // 2]
    df2 = df.copy()
    df2.loc[df2["target"] >= o, "y"] += 5000
    a = md.combine(df, "good", "bad", "c", weights="adaptive")
    b = md.combine(df2, "good", "bad", "c", weights="adaptive")
    assert np.allclose(a.loc[a["origin"] == o, "point"], b.loc[b["origin"] == o, "point"])


def test_level_aware_intervals_follow_the_level():
    rng = np.random.default_rng(9)
    idx = pd.date_range("2025-01-01", periods=400, freq="D")
    lev = np.where(np.arange(400) < 300, 50.0, 500.0)
    y = pd.Series(rng.gamma(4.0, lev / 4), index=idx)
    rows = []
    for o in idx[1:-7]:
        tg = pd.date_range(o, periods=7)
        p = float(lev[idx.get_loc(o) - 1])
        rows.append(pd.DataFrame({"origin": o, "h": range(1, 8), "target": tg, "method": "m", "y": y.reindex(tg).to_numpy(), "point": p}))
    df = pd.concat(rows, ignore_index=True)
    a = md.add_intervals(df, mode="level", lookback=365)
    b = md.add_intervals(df, mode="recent", lookback=30)
    late = a["origin"] >= idx[340]
    assert a.loc[late, "q0.9"].notna().all() and np.allclose(a.loc[late, "q0.1"].to_numpy() <= a.loc[late, "q0.9"].to_numpy(), True)
    e = md.recent_errors(df, "m", idx[350], mode="level", point_now=500.0, lookback=365)
    assert len(e) >= 28


def test_combine_weight_matches_combine():
    df = _toy()
    ad = md.combine(df, "good", "bad", "c", weights="adaptive", lookback=30)
    o = ad["origin"].iloc[-1]
    w = md.combine_weight(df, "good", "bad", o, weights="adaptive", lookback=30)
    assert np.isclose(w, ad.loc[ad["origin"] == o, "weight_a"].iloc[0])
