import datetime as dt
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import weather as w  # noqa: E402


class FakeResp:
    def __init__(self, code, data=None, text=""):
        self.status_code, self._d, self.text = code, data, text

    def json(self):
        return self._d


def fake_get_factory(reject_extra=False, last_hist_day=None):
    calls = []

    def get(url, params=None, timeout=None):
        calls.append((url, params))
        vars_ = params["daily"].split(",")
        if reject_extra and "cloud_cover_mean" in vars_:
            return FakeResp(400, text='{"error":true,"reason":"bad variable"}')
        days = pd.date_range(params["start_date"], params["end_date"], freq="D")
        daily = {"time": [d.strftime("%Y-%m-%d") for d in days]}
        for v in vars_:
            if v in ("sunrise", "sunset"):
                daily[v] = [d.strftime("%Y-%m-%dT06:00") for d in days]
            elif v == "daylight_duration":
                daily[v] = [36000.0] * len(days)
            elif v == "sunshine_duration":
                daily[v] = [18000.0] * len(days)
            else:
                daily[v] = [5.0] * len(days)
        if last_hist_day and "historical" in url:   # archive lagging: nulls after last_hist_day
            for i, d in enumerate(days):
                if d.date() > last_hist_day:
                    for v in vars_:
                        if v not in ("sunrise", "sunset"):
                            daily[v][i] = None
        return FakeResp(200, {"daily": daily})

    get.calls = calls
    return get


def test_build_full_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "RAW_DIR", tmp_path)
    get = fake_get_factory()
    today = dt.date(2026, 10, 12)
    df, end = w.build(dt.date(2022, 7, 1), dt.date(2026, 10, 13), 7, get=get, today=today)
    assert end == dt.date(2026, 10, 19)
    assert not [r for r in w.check_frame(df, dt.date(2022, 7, 1), end) if not r[1]]
    assert (df.source == "forecast").sum() >= 8            # today..end come from the live forecast (+ week back fill not needed)
    assert df["date"].is_unique and df["daylight_min"].iloc[0] == 600.0
    # five yearly archive calls (2022..2026) + one live call
    assert len(get.calls) == 6
    assert len(list(tmp_path.glob("*.json"))) == 6


def test_archive_lag_is_filled_from_live_forecast(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "RAW_DIR", tmp_path)
    today = dt.date(2026, 10, 12)
    get = fake_get_factory(last_hist_day=dt.date(2026, 10, 9))
    df, end = w.build(dt.date(2026, 9, 1), dt.date(2026, 10, 13), 3, get=get, today=today)
    assert not [r for r in w.check_frame(df, dt.date(2026, 9, 1), end) if not r[1]]
    row = df[df["date"] == dt.date(2026, 10, 10)].iloc[0]
    assert row["source"] == "forecast"


def test_optional_variables_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "RAW_DIR", tmp_path)
    get = fake_get_factory(reject_extra=True)
    df, _ = w.build(dt.date(2026, 9, 1), dt.date(2026, 10, 13), 2, get=get, today=dt.date(2026, 10, 12))
    assert "cloud_cover_mean" not in df.columns and "temperature_2m_mean" in df.columns


def test_too_far_ahead_is_refused():
    try:
        w.build(dt.date(2026, 9, 1), dt.date(2026, 12, 1), 7, get=fake_get_factory(), today=dt.date(2026, 10, 12), save_raw=False)
    except ValueError:
        return
    raise AssertionError("should refuse an end date beyond the live forecast reach")


def test_check_detects_gap():
    df = pd.DataFrame({"date": [dt.date(2026, 1, 1), dt.date(2026, 1, 3)], "temperature_2m_mean": [1.0, 2.0],
                       "source": "x", "fetched_utc": "x"})
    res = dict((n, ok) for n, ok, _ in w.check_frame(df, dt.date(2026, 1, 1), dt.date(2026, 1, 3)))
    assert res["every day present"] is False
