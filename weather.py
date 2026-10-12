"""Daily weather for Kyiv from Open-Meteo, used as extra (non-alert) data for the
minutes-under-alert forecast.

Two Open-Meteo endpoints are combined into one table with one row per local day:

* Historical Forecast API (archived forecasts): from the start of our alert series up to yesterday.
* Weather Forecast API (live forecast): today up to the end of the forecast horizon.

IMPORTANT, read before using the numbers as features
----------------------------------------------------
The archived series is stitched together from the first hours of successive model runs, so the value
for day D is (roughly) a forecast issued on day D itself, not a forecast issued one to seven days
earlier. For the day-1 target this is close to what we would have had at the forecast origin; for days
2 to 7 it is *better* than what we would really have known, so backtest gains from weather at longer
horizons are optimistic. The Single Runs API would give the exact lead time; it is not used here.
Model changes over time also make the archived series slightly inconsistent.

Usage (from the project folder):
    uv run python weather.py fetch                       # start 2022-07-01, origin = tomorrow, 7 days
    uv run python weather.py fetch --origin 2026-10-13   # fixed forecast origin (first forecast day)
    uv run python weather.py check weather_kyiv_daily.csv

Terms: Open-Meteo is free for non-commercial use (CC BY 4.0, attribute "Weather data by Open-Meteo.com").
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "weather_raw"
DEFAULT_OUT = HERE / "weather_kyiv_daily.csv"

LAT, LON, TZ = 50.45, 30.52, "Europe/Kyiv"
HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FCST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_START = "2022-07-01"   # first day of the usable alert series (README L9)
DEFAULT_HORIZON = 7            # forecast days after the origin
MAX_FORECAST_AHEAD = 15        # the live API serves at most 16 days including today

# Variables that exist on both endpoints.
CORE = [
    "weather_code",
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "apparent_temperature_mean",
    "sunrise", "sunset", "daylight_duration", "sunshine_duration",
    "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
]
# Nice to have; if the API rejects any of them the run falls back to CORE and says so.
EXTRA = ["cloud_cover_mean", "relative_humidity_2m_mean", "pressure_msl_mean", "visibility_mean"]


def kyiv_today() -> dt.date:
    return dt.datetime.now(ZoneInfo(TZ)).date()


def _get_json(url: str, params: dict, get=requests.get, tries: int = 3) -> dict:
    last = None
    for i in range(tries):
        r = get(url, params=params, timeout=90)
        if r.status_code == 200:
            return r.json()
        last = r
        if r.status_code in (429, 500, 502, 503, 504) and i < tries - 1:
            time.sleep(5 * (i + 1))
            continue
        break
    raise RuntimeError(f"HTTP {last.status_code} from {url}: {last.text[:300]}")


def _params(variables, start, end, extra=None):
    p = {"latitude": LAT, "longitude": LON, "daily": ",".join(variables), "timezone": TZ,
         "start_date": str(start), "end_date": str(end)}
    p.update(extra or {})
    return p


def fetch_payloads(url: str, start: dt.date, end: dt.date, get=requests.get, chunk_by_year=True):
    """Return a list of JSON payloads covering start..end (one call per calendar year)."""
    spans, s = [], start
    while s <= end:
        e = min(end, dt.date(s.year, 12, 31)) if chunk_by_year else end
        spans.append((s, e))
        s = e + dt.timedelta(days=1)
    out = []
    for s, e in spans:
        try:
            out.append(_get_json(url, _params(CORE + EXTRA, s, e), get))
        except RuntimeError as err:
            if "HTTP 400" not in str(err):
                raise
            print(f"warning: optional variables rejected ({str(err)[:200]}); retrying with core variables only",
                  file=sys.stderr)
            out.append(_get_json(url, _params(CORE, s, e), get))
    return out


def frame_from_payload(payload: dict, source: str, fetched_utc: str) -> pd.DataFrame:
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        raise ValueError("payload has no daily block")
    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ("sunrise", "sunset"):
        if c in df:
            df[c] = pd.to_datetime(df[c])          # local time, no offset (we asked for Europe/Kyiv)
    if "daylight_duration" in df:                  # seconds -> minutes
        df["daylight_min"] = df.pop("daylight_duration") / 60.0
    if "sunshine_duration" in df:
        df["sunshine_min"] = df.pop("sunshine_duration") / 60.0
    df["source"] = source
    df["fetched_utc"] = fetched_utc
    return df


def combine(hist: pd.DataFrame, fcst: pd.DataFrame) -> pd.DataFrame:
    """One row per date. Archived forecast wins when it has data; otherwise the live forecast fills in."""
    key = "temperature_2m_mean"
    hist_ok = hist[hist[key].notna()] if len(hist) else hist
    have = set(hist_ok["date"]) if len(hist_ok) else set()
    fill = fcst[~fcst["date"].isin(have)]
    out = pd.concat([hist_ok, fill], ignore_index=True)
    out = out.drop_duplicates("date", keep="first").sort_values("date").reset_index(drop=True)
    return out


def check_frame(df: pd.DataFrame, start: dt.date, end: dt.date) -> list[tuple[str, bool, str]]:
    res = []
    expected = pd.date_range(start, end, freq="D").date
    missing = sorted(set(expected) - set(df["date"]))
    res.append(("every day present", not missing, f"{len(missing)} missing" + (f", first {missing[0]}" if missing else "")))
    res.append(("no duplicate days", df["date"].is_unique, ""))
    t = df["temperature_2m_mean"]
    res.append(("temperature plausible (-45..45 C)", bool(t.dropna().between(-45, 45).all()), f"min {t.min():.1f}, max {t.max():.1f}"))
    res.append(("no missing temperature", bool(t.notna().all()), f"{int(t.isna().sum())} missing"))
    if "daylight_min" in df:
        d = df["daylight_min"].dropna()
        res.append(("daylight plausible (200..1300 min)", bool(d.between(200, 1300).all()), f"min {d.min():.0f}, max {d.max():.0f}"))
    if "precipitation_sum" in df:
        res.append(("precipitation not negative", bool((df["precipitation_sum"].dropna() >= 0).all()), ""))
    nulls = df.drop(columns=["source", "fetched_utc"]).isna().sum()
    nulls = nulls[nulls > 0]
    res.append(("columns with gaps", True, ", ".join(f"{k}={v}" for k, v in nulls.items()) or "none"))
    return res


def build(start: dt.date, origin: dt.date, horizon: int, get=requests.get, today: dt.date | None = None, save_raw=True):
    today = today or kyiv_today()
    end = origin + dt.timedelta(days=horizon - 1)
    if (end - today).days > MAX_FORECAST_AHEAD:
        raise ValueError(f"end date {end} is more than {MAX_FORECAST_AHEAD} days ahead; the live forecast cannot reach it")
    fetched = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = fetched.replace("-", "").replace(":", "")
    hist_end = min(end, today - dt.timedelta(days=1))
    hist_payloads = fetch_payloads(HIST_URL, start, hist_end, get)
    # live forecast: from a week back (to fill any lag in the archive) to the end of the horizon
    f_start = max(start, today - dt.timedelta(days=7))
    fcst_payloads = fetch_payloads(FCST_URL, f_start, end, get, chunk_by_year=False)
    if save_raw:
        RAW_DIR.mkdir(exist_ok=True)
        for tag, ps in (("historical_forecast", hist_payloads), ("forecast", fcst_payloads)):
            for i, p in enumerate(ps):
                (RAW_DIR / f"{tag}_{stamp}_{i}.json").write_text(json.dumps(p), encoding="utf-8")
    hist = pd.concat([frame_from_payload(p, "historical_forecast", fetched) for p in hist_payloads], ignore_index=True)
    fcst = pd.concat([frame_from_payload(p, "forecast", fetched) for p in fcst_payloads], ignore_index=True)
    df = combine(hist, fcst)
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    return df, end


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download and write the daily weather table")
    f.add_argument("--start", default=DEFAULT_START, help="first day (default %(default)s)")
    f.add_argument("--origin", help="first forecast day, YYYY-MM-DD (default: tomorrow, Kyiv time)")
    f.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="forecast days from the origin (default %(default)s)")
    f.add_argument("--out", default=str(DEFAULT_OUT))
    c = sub.add_parser("check", help="sanity checks on a written CSV")
    c.add_argument("csv")
    a = ap.parse_args(argv)

    if a.cmd == "fetch":
        start = dt.date.fromisoformat(a.start)
        origin = dt.date.fromisoformat(a.origin) if a.origin else kyiv_today() + dt.timedelta(days=1)
        try:
            df, end = build(start, origin, a.horizon)
        except (RuntimeError, ValueError, requests.RequestException) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        df.to_csv(a.out, index=False)
        first_hist = df.loc[df.source == "historical_forecast", "date"].min()
        print(f"wrote {a.out}: {len(df)} days {df['date'].min()} .. {df['date'].max()} "
              f"({(df.source == 'historical_forecast').sum()} archived, {(df.source == 'forecast').sum()} live); "
              f"archive starts {first_hist}")
        res = check_frame(df, start, end)
    else:
        df = pd.read_csv(a.csv, parse_dates=["sunrise", "sunset"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        res = check_frame(df, df["date"].min(), df["date"].max())

    bad = 0
    for name, ok, info in res:
        print(("PASS " if ok else "FAIL ") + name + (f": {info}" if info else ""))
        bad += (not ok)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
