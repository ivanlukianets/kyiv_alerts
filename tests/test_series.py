import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import series as sr  # noqa: E402


def mk(rows):
    """rows of (start, end) as UTC strings; end None = still running."""
    return pd.DataFrame({"started_at": pd.to_datetime([r[0] for r in rows], utc=True),
                         "finished_at": pd.to_datetime([r[1] for r in rows], utc=True)})


def test_midnight_split_and_local_days():
    # 21:30 UTC on 11 Oct = 00:30 Kyiv on 12 Oct (UTC+3). Alert 21:00-22:00 UTC = 00:00-01:00 local on 12 Oct
    a = mk([("2026-10-11 20:30", "2026-10-11 21:30")])      # 23:30 Kyiv 11 Oct .. 00:30 Kyiv 12 Oct
    d = sr.minutes_by_bin(a, "D", as_of="2026-10-13")
    assert d.loc["2026-10-11", "minutes"] == 30 and d.loc["2026-10-12", "minutes"] == 30
    assert d["minutes"].sum() == 60 and d["n_starts"].sum() == 1
    assert d.loc["2026-10-11", "n_starts"] == 1          # counted on the day it started


def test_clock_change_day_has_25_hours():
    a = mk([("2026-10-24 21:00", "2026-10-24 22:00")])
    d = sr.minutes_by_bin(a, "D", start="2026-10-23", end="2026-10-27", as_of="2026-10-28")
    lengths = d["bin_minutes"].to_dict()
    assert lengths[pd.Timestamp("2026-10-25", tz=sr.TZ)] == 1500
    assert lengths[pd.Timestamp("2026-10-24", tz=sr.TZ)] == 1440


def test_hourly_and_daily_agree_and_total_matches():
    a = mk([("2026-03-01 10:15", "2026-03-01 11:45"), ("2026-03-02 22:30", "2026-03-03 01:10"),
            ("2026-03-05 05:00", "2026-03-05 05:07")])
    total = (a["finished_at"] - a["started_at"]).dt.total_seconds().sum() / 60
    asof = "2026-03-10"
    h = sr.minutes_by_bin(a, "h", as_of=asof)
    d = sr.minutes_by_bin(a, "D", as_of=asof)
    q = sr.minutes_by_bin(a, "15min", as_of=asof)
    assert abs(h["minutes"].sum() - total) < 1e-6 and abs(d["minutes"].sum() - total) < 1e-6 and abs(q["minutes"].sum() - total) < 1e-6
    assert (h["minutes"] <= h["bin_minutes"] + 1e-9).all()
    # regrouping hourly to local days gives the daily table
    regroup = h["minutes"].groupby(h.index.tz_convert(sr.TZ).normalize()).sum()
    assert np.allclose(regroup.reindex(d.index).fillna(0).to_numpy(), d["minutes"].to_numpy())


def test_open_alert_counts_only_until_as_of():
    a = mk([("2026-10-12 08:00", None)])
    h = sr.minutes_by_bin(a, "h", as_of="2026-10-12 09:30")
    assert abs(h["minutes"].sum() - 90) < 1e-9
    assert h["complete"].sum() == 1 and not h["complete"].iloc[-1]    # the 09:00 hour is still running


def test_overlapping_alerts_are_merged():
    a = mk([("2026-10-12 08:00", "2026-10-12 09:00"), ("2026-10-12 08:30", "2026-10-12 09:30")])
    assert abs(sr.minutes_by_bin(a, "h", as_of="2026-10-13")["minutes"].sum() - 90) < 1e-9


def test_truncation_no_future_leak():
    """Bins finished before t are identical whether or not data after t exists."""
    a = mk([("2026-03-01 10:15", "2026-03-01 11:45"), ("2026-03-02 22:30", "2026-03-03 01:10"),
            ("2026-03-04 05:00", "2026-03-04 05:07"), ("2026-03-06 12:00", "2026-03-06 14:00")])
    t = pd.Timestamp("2026-03-03 00:30", tz="UTC")               # in the middle of the second alert
    full = sr.minutes_by_bin(a, "h", as_of="2026-03-10", start="2026-03-01", end="2026-03-07")
    cut = sr.minutes_by_bin(sr.as_of_view(a, t), "h", as_of=t, start="2026-03-01", end="2026-03-07")
    done = cut[cut["complete"]]
    assert len(done) > 40
    assert np.allclose(done["minutes"], full.loc[done.index, "minutes"])
    assert (cut.loc[~cut["complete"], "minutes"] <= 60).all()
    view = sr.as_of_view(a, t)
    assert len(view) == 2 and view["finished_at"].isna().iloc[1]   # the running alert's end is hidden


def test_empty_and_unfinished_only():
    a = mk([("2026-10-12 08:00", None)])
    d = sr.minutes_by_bin(a, "D", as_of="2026-10-12 08:00")
    assert d["minutes"].sum() == 0


def test_old_alert_without_end_is_not_stretched():
    """Regression: a gap in the source (start without all-clear) must not run to as_of."""
    a = mk([("2026-10-01 08:00", None), ("2026-10-05 08:00", "2026-10-05 09:00"), ("2026-10-12 08:00", None)])
    d = sr.minutes_by_bin(a, "h", as_of="2026-10-12 09:00")
    assert abs(d["minutes"].sum() - (60 + 60)) < 1e-9          # the 5 Oct alert + the running one; 1 Oct ignored
    assert d["n_starts"].sum() == 3
