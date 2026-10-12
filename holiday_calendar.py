"""Calendar of holidays and symbolic dates (Ukraine, Russia, shared), built by hand from rules.

Not named `holidays.py` on purpose: that would shadow the third-party `holidays` package.

Two outputs, both small and deterministic (no network):

* holiday_events.csv          one row per date and event, long format:
                              date, country, name, kind, movable, rule
* holiday_features_daily.csv  one row per day, for every group of kinds:
                              <group>_is_day, _eve, _within2, _days_to_next, _days_since_last

Usage:
    uv run python holiday_calendar.py build                     # 2022-07-01 .. 2026-12-31
    uv run python holiday_calendar.py build --start 2022-07-01 --end 2027-03-31

Kinds (they are also the feature groups):
    ua_statehood         Ukrainian statehood dates
    ua_military          Ukrainian military and remembrance dates
    ru_state_military    Russian state and military dates
    religious            Easter, Trinity, Christmas (country says whose calendar)
    new_year             31 Dec and 1 Jan
    war_anniversary      24 Feb
    civil                other civil dates; listed in the events table, not turned into features

IMPORTANT: this is a hand-written table, not an official source. Check it before relying on it.
Whether a date is a day off is deliberately not recorded (Ukrainian days off were changed by martial law
and we have not verified them). A date being "symbolic" is not the same as being a day off.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EVENTS_CSV = HERE / "holiday_events.csv"
FEATURES_CSV = HERE / "holiday_features_daily.csv"
CAP = 14                      # days_to_next / days_since_last are capped here
GROUPS = ["ua_statehood", "ua_military", "ru_state_military", "religious", "new_year", "war_anniversary"]


def orthodox_easter(year: int) -> dt.date:
    """Orthodox Easter as a Gregorian date (Meeus Julian algorithm, valid 1900-2099)."""
    a, b, c = year % 4, year % 7, year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month, day = divmod(d + e + 114, 31)
    return dt.date(year, month, day + 1) + dt.timedelta(days=13)


def last_sunday(year: int, month: int) -> dt.date:
    d = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


# (country, name, kind, rule, first_year, last_year); rule is "MM-DD", "easter+N", or "last_sun_M"
RULES = [
    # Ukrainian statehood
    ("UA", "Unity Day", "ua_statehood", "01-22", None, None),
    ("UA", "Flag Day", "ua_statehood", "08-23", None, None),
    ("UA", "Independence Day", "ua_statehood", "08-24", None, None),
    ("UA", "Constitution Day", "ua_statehood", "06-28", None, None),
    ("UA", "Statehood Day", "ua_statehood", "07-15", 2022, None),
    # Ukrainian military and remembrance
    ("UA", "Defenders Day (old date)", "ua_military", "10-14", None, 2022),
    ("UA", "Defenders Day", "ua_military", "10-01", 2023, None),
    ("UA", "Armed Forces Day", "ua_military", "12-06", None, None),
    ("UA", "Dignity and Freedom Day", "ua_military", "11-21", None, None),
    ("UA", "Day of Remembrance and Victory over Nazism", "ua_military", "05-08", 2023, None),
    # Russian state and military
    ("RU", "Defender of the Fatherland Day", "ru_state_military", "02-23", None, None),
    ("RU", "Victory Day", "ru_state_military", "05-09", None, None),
    ("RU", "Russia Day", "ru_state_military", "06-12", None, None),
    ("RU", "Navy Day", "ru_state_military", "last_sun_7", None, None),
    ("RU", "National Unity Day", "ru_state_military", "11-04", None, None),
    ("RU", "Constitution Day (Russia)", "ru_state_military", "12-12", None, None),
    # religious
    ("both", "Orthodox Easter", "religious", "easter+0", None, None),
    ("both", "Trinity (Pentecost)", "religious", "easter+49", None, None),
    ("RU", "Christmas (7 Jan)", "religious", "01-07", None, None),
    ("UA", "Christmas (7 Jan, last time official)", "religious", "01-07", 2023, 2023),
    ("UA", "Christmas (25 Dec)", "religious", "12-25", 2023, None),
    # new year
    ("both", "New Year's Eve", "new_year", "12-31", None, None),
    ("both", "New Year's Day", "new_year", "01-01", None, None),
    # war anniversary
    ("both", "Anniversary of the full-scale invasion", "war_anniversary", "02-24", 2023, None),
    # other civil dates (events table only)
    ("both", "International Women's Day", "civil", "03-08", None, None),
    ("both", "Labour Day", "civil", "05-01", None, None),
    ("UA", "Kyiv Day", "civil", "last_sun_5", None, None),
]


def resolve(rule: str, year: int) -> dt.date:
    if rule.startswith("easter+"):
        return orthodox_easter(year) + dt.timedelta(days=int(rule.split("+")[1]))
    if rule.startswith("last_sun_"):
        return last_sunday(year, int(rule.split("_")[2]))
    m, d = rule.split("-")
    return dt.date(year, int(m), int(d))


def build_events(first_year: int, last_year: int) -> pd.DataFrame:
    rows = []
    for country, name, kind, rule, y0, y1 in RULES:
        for y in range(first_year, last_year + 1):
            if (y0 and y < y0) or (y1 and y > y1):
                continue
            rows.append({"date": resolve(rule, y), "country": country, "name": name, "kind": kind,
                         "movable": rule.startswith(("easter", "last_sun")), "rule": rule})
    return pd.DataFrame(rows).sort_values(["date", "country", "name"]).reset_index(drop=True)


def build_features(events: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    out = pd.DataFrame({"date": days.date})
    for g in GROUPS:
        ev = pd.to_datetime(events.loc[events["kind"] == g, "date"]).drop_duplicates().sort_values()
        to_next, since = [], []
        for d in days:
            after = ev[ev >= d]
            before = ev[ev <= d]
            to_next.append(min((after.iloc[0] - d).days, CAP) if len(after) else CAP)
            since.append(min((d - before.iloc[-1]).days, CAP) if len(before) else CAP)
        out[f"{g}_days_to_next"] = to_next
        out[f"{g}_days_since_last"] = since
        out[f"{g}_is_day"] = (out[f"{g}_days_to_next"] == 0).astype(int)
        out[f"{g}_eve"] = (out[f"{g}_days_to_next"] == 1).astype(int)
        out[f"{g}_within2"] = ((out[f"{g}_days_to_next"] <= 2) | (out[f"{g}_days_since_last"] <= 2)).astype(int)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--start", default="2022-07-01")
    b.add_argument("--end", default="2026-12-31")
    b.add_argument("--events-out", default=str(EVENTS_CSV))
    b.add_argument("--features-out", default=str(FEATURES_CSV))
    a = ap.parse_args(argv)
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    if end < start:
        print("error: --end is before --start", file=sys.stderr)
        return 2
    events = build_events(start.year - 1, end.year + 1)     # margin so days_since/to_next work at the edges
    feats = build_features(events, start, end)
    inside = events[(events["date"] >= start) & (events["date"] <= end)]
    inside.to_csv(a.events_out, index=False)
    feats.to_csv(a.features_out, index=False)
    print(f"wrote {a.events_out}: {len(inside)} events {start}..{end}")
    print(f"wrote {a.features_out}: {len(feats)} days x {feats.shape[1] - 1} features")
    per = inside.groupby("kind").size().to_dict()
    print("events per kind:", per)
    return 0


if __name__ == "__main__":
    sys.exit(main())
