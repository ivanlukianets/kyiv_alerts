import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import explore as ex  # noqa: E402


def test_change_point_found_near_the_step():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(5, 1, 300), rng.normal(15, 1, 300)])
    cps = ex.change_points(x)
    assert len(cps) == 1 and abs(cps[0] - 300) <= 5


def test_no_change_point_in_noise():
    x = np.random.default_rng(1).normal(5, 1, 600)
    assert ex.change_points(x) == []


def test_acf_of_ar1_and_detrend():
    rng = np.random.default_rng(2)
    e = rng.normal(size=4000)
    x = np.zeros(4000)
    for i in range(1, 4000):
        x[i] = 0.7 * x[i - 1] + e[i]
    a = ex.acf(pd.Series(x), 3)
    assert abs(a[1] - 0.7) < 0.05 and abs(a[2] - 0.49) < 0.06
    # pure level shift looks like persistence until the level is removed
    step = pd.Series(np.concatenate([rng.normal(0, 1, 500), rng.normal(10, 1, 500)]))
    assert ex.acf(step, 1)[1] > 0.5
    assert ex.acf(step, 1, detrend_window=31)[1] < 0.2


def test_heavy_lift_detects_clustering():
    rng = np.random.default_rng(3)
    iid = pd.Series(rng.gamma(1.0, 50, 5000))
    assert abs(ex.heavy_lift(iid, 0.9, [1])["lift"][0] - 1) < 0.35
    clustered = iid.copy()
    for s in range(0, 5000, 100):                       # a run of 10 very heavy days every 100 days
        clustered.iloc[s:s + 10] += 500
    assert ex.heavy_lift(clustered, 0.9, [1])["lift"][0] > 5


def test_dispersion_slope():
    rng = np.random.default_rng(4)
    levels = np.repeat(np.exp(rng.uniform(np.log(10), np.log(300), 80)), 30)
    mult = pd.Series(rng.gamma(4, levels / 4))           # std proportional to level
    const = pd.Series(levels + rng.normal(0, 20, len(levels)))   # std independent of level
    assert 0.8 < ex.dispersion_blocks(mult, 30)[1] < 1.2
    assert abs(ex.dispersion_blocks(const, 30)[1]) < 0.3
