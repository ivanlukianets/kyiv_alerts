import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import holiday_calendar as hc  # noqa: E402


def test_orthodox_easter_known_dates():
    known = {2022: (4, 24), 2023: (4, 16), 2024: (5, 5), 2025: (4, 20), 2026: (4, 12)}
    for y, (m, d) in known.items():
        assert hc.orthodox_easter(y) == dt.date(y, m, d), y


def test_last_sunday():
    assert hc.last_sunday(2026, 7) == dt.date(2026, 7, 26)
    assert hc.last_sunday(2026, 5) == dt.date(2026, 5, 31)
    assert hc.last_sunday(2026, 12) == dt.date(2026, 12, 27)


def test_year_bounds_and_movable():
    ev = hc.build_events(2022, 2026)
    dd = ev[ev["name"].str.startswith("Defenders Day")]
    assert dt.date(2022, 10, 14) in set(dd["date"]) and dt.date(2022, 10, 1) not in set(dd["date"])
    assert dt.date(2023, 10, 1) in set(dd["date"]) and dt.date(2023, 10, 14) not in set(dd["date"])
    tr = ev[ev["name"].str.startswith("Trinity")]
    assert dt.date(2023, 6, 4) in set(tr["date"])
    assert dt.date(2022, 12, 25) not in set(ev[ev["name"] == "Christmas (25 Dec)"]["date"])


def test_features(tmp_path):
    ev = hc.build_events(2022, 2027)
    f = hc.build_features(ev, dt.date(2026, 8, 20), dt.date(2026, 8, 28)).set_index("date")
    assert f.loc[dt.date(2026, 8, 24), "ua_statehood_is_day"] == 1
    assert f.loc[dt.date(2026, 8, 23), "ua_statehood_is_day"] == 1       # Flag Day is its own event
    assert f.loc[dt.date(2026, 8, 22), "ua_statehood_eve"] == 1
    assert f.loc[dt.date(2026, 8, 26), "ua_statehood_within2"] == 1
    assert f.loc[dt.date(2026, 8, 28), "ua_statehood_within2"] == 0
    assert f.loc[dt.date(2026, 8, 24), "ua_statehood_days_since_last"] == 0
    assert (f.filter(like="days_to_next").max() <= hc.CAP).all()


def test_cli(tmp_path):
    e, f = tmp_path / "e.csv", tmp_path / "f.csv"
    assert hc.main(["build", "--start", "2026-10-01", "--end", "2026-10-31", "--events-out", str(e), "--features-out", str(f)]) == 0
    assert "Defenders Day" in e.read_text(encoding="utf-8")
    assert hc.main(["build", "--start", "2026-10-02", "--end", "2026-10-01"]) == 2
