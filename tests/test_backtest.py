import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backtest as bt  # noqa: E402


def mk(n=120, start="2024-01-01", seed=0):
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(np.random.default_rng(seed).gamma(2.0, 30, n).round(), index=idx)


def test_simple_methods_by_hand():
    y = pd.Series(np.arange(1.0, 29.0), index=pd.date_range("2024-01-01", periods=28))
    o = y.index[-1] + pd.Timedelta(days=1)
    f = lambda spec: bt.forecast(y, o, 7, spec)[0]
    assert (f(("last",)) == 28).all()
    assert (f(("mean", 7)) == y.iloc[-7:].mean()).all()
    assert (f(("median", 3)) == 27).all()
    assert list(f(("seasonal",))) == [22, 23, 24, 25, 26, 27, 28]              # 7 days back from each target
    assert list(f(("wd4",))) == [np.mean([22, 15, 8, 1]) + i for i in range(7)]
    e = f(("ewma", 1))
    w = 0.5 ** np.arange(28)[::-1]
    assert np.isclose(e[0], np.average(y, weights=w))


def test_median_methods_give_ordered_quantiles():
    y = mk()
    _, q = bt.forecast(y, y.index[-1] + pd.Timedelta(days=1), 7, ("median", 60))
    assert q.shape == (7, 5) and (np.diff(q, axis=1) >= 0).all()


def test_backtest_cannot_see_the_future():
    y = mk(200)
    origins = pd.date_range("2024-04-01", "2024-06-01", freq="7D")
    m = {k: v for k, v in bt.default_methods().items() if k in
         ("last", "same_weekday", "mean_30", "median_all", "ewma_14", "since_break_mean")}
    a = bt.run_backtest(y, origins, 7, m)
    y2 = y.copy()
    y2.loc[y2.index >= origins[3]] += 1000                       # change everything from the 4th origin on
    b = bt.run_backtest(y2, origins, 7, m)
    first = a["origin"] < origins[3]
    cols = ["point"] + [f"q{q}" for q in bt.QS]
    assert first.sum() > 0
    pd.testing.assert_frame_equal(a.loc[first, cols].reset_index(drop=True), b.loc[b["origin"] < origins[3], cols].reset_index(drop=True))
    assert not np.allclose(a["point"], b["point"])               # but later origins do change


def test_origins_without_full_targets_are_skipped():
    y = mk(60)
    out = bt.run_backtest(y, [y.index[-3], y.index[-7]], 7, {"last": ("last",)})
    assert set(out["origin"]) == {y.index[-7]}


def test_score_and_interval_score():
    d = pd.DataFrame({"method": ["a"] * 4, "y": [10.0, 10, 10, 10], "point": [12.0, 8, 10, 14],
                      "q0.1": [5.0] * 4, "q0.9": [11.0] * 4, "q0.025": [0.0] * 4, "q0.975": [20.0] * 4})
    s = bt.score(d, scale=2.0)
    assert s.loc["a", "mae"] == 2 and s.loc["a", "mase"] == 1 and s.loc["a", "bias"] == 1
    iv = bt.interval_score(d)
    assert iv.loc["a", "cov80"] == 1 and iv.loc["a", "cov95"] == 1 and iv.loc["a", "width80"] == 6
    d.loc[0, "y"] = 20.0                                          # 9 above the 80% band
    assert bt.interval_score(d).loc["a", "cov80"] == 0.75


def test_iid_coverage_is_near_nominal():
    y = mk(900, seed=5)
    out = bt.run_backtest(y, pd.date_range("2025-01-01", "2026-04-01", freq="D"), 7, {"median_365": ("median", 365)})
    iv = bt.interval_score(out).iloc[0]
    assert abs(iv["cov80"] - 0.8) < 0.05 and abs(iv["cov95"] - 0.95) < 0.04


def test_dm_test():
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 500)
    _, _, p_same = bt.dm_test(a + rng.normal(0, 1, 500), a + rng.normal(0, 1, 500))
    assert p_same > 0.01
    mu, stat, p = bt.dm_test(a + 0.5, a)
    assert abs(mu - 0.5) < 1e-9 and p < 1e-6
    # positive autocorrelation makes the error bar wider than the naive one
    e = rng.normal(0, 1, 2000)
    ar = np.zeros(2000)
    for i in range(1, 2000):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    _, stat_hac, _ = bt.dm_test(ar, np.zeros(2000), lags=10)
    naive = ar.mean() / (ar.std() / np.sqrt(2000))
    assert abs(stat_hac) < abs(naive)
