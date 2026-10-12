"""Turn alert intervals into minutes under alert per time bin (hourly, daily, 15 minutes, ...).

One function for every granularity, so hourly and daily numbers cannot disagree.

    alerts = ...                      # DataFrame with UTC columns started_at, finished_at (NaT = still running)
    daily  = minutes_by_bin(alerts, "D", as_of=snapshot_time)
    hourly = minutes_by_bin(alerts, "h", as_of=snapshot_time)

Rules (they matter for leakage, so they are tested in tests/test_series.py):
* An alert is clipped at the bin edges: an alert across midnight counts in both days.
* Daily bins are LOCAL calendar days (Europe/Kyiv), so a clock-change day has 23 or 25 hours.
  Other frequencies (h, 15min, ...) are aligned to UTC, which equals local time up to whole hours in Kyiv.
* The newest alert, if it has no end, counts only up to `as_of` (the time of the snapshot, or the forecast origin).
  An OLDER alert without an end has an unknown length and is ignored for minutes (it still counts as a start).
  Nothing is ever given a made-up end.
* `as_of_view(alerts, t)` is what a forecaster standing at time t may know: alerts that started before t,
  with every end after t hidden. Features must be built from this view only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "Europe/Kyiv"
_NS_PER_MIN = 60_000_000_000


def _ns(x) -> np.ndarray:
    """UTC nanoseconds since the epoch as int64, whatever resolution pandas stores (ns, us, ...)."""
    idx = pd.DatetimeIndex(x)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.tz_localize(None).as_unit("ns").asi8


def read_alerts_csv(path) -> pd.DataFrame:
    """Read the CSV written by `kyiv_digital.py parse` (UTC timestamps)."""
    df = pd.read_csv(path)
    for c in ("started_at", "finished_at", "snapshot_utc"):
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True)
    return df


def as_of_view(alerts: pd.DataFrame, t) -> pd.DataFrame:
    """Alerts as known at time t: started before t, ends after t hidden (NaT)."""
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    out = alerts[alerts["started_at"] < t].copy()
    hidden = out["finished_at"].isna() | (out["finished_at"] > t)
    out.loc[hidden, "finished_at"] = pd.NaT
    return out.reset_index(drop=True)


def _merged_intervals(alerts: pd.DataFrame, as_of) -> tuple[np.ndarray, np.ndarray]:
    """Sorted, non-overlapping [start, end) as int64 ns. Open alerts end at as_of."""
    s = alerts["started_at"]
    e = alerts["finished_at"].copy()
    if as_of is not None:
        # Only the newest alert may still be running. An older alert without an all-clear (a gap in the source,
        # mostly early 2022) has an unknown length: it is ignored here, never stretched to `as_of`.
        newest = s == s.max()
        e = e.where(~(e.isna() & newest), as_of)
    keep = e.notna()
    s, e = s[keep], e[keep]
    if as_of is not None:                              # nothing may extend past what is known
        e = e.clip(upper=as_of)
        s = s.clip(upper=as_of)
    s, e = _ns(s), _ns(e)
    order = np.argsort(s)
    s, e = s[order], e[order]
    ms, me = [], []
    for a, b in zip(s, e):
        if b <= a:
            continue
        if ms and a <= me[-1]:
            me[-1] = max(me[-1], b)
        else:
            ms.append(a)
            me.append(b)
    return np.array(ms, dtype="int64"), np.array(me, dtype="int64")


def _cumulative(starts: np.ndarray, ends: np.ndarray, t: np.ndarray) -> np.ndarray:
    """F(t): alert nanoseconds accumulated up to each time t (merged intervals)."""
    if len(starts) == 0:
        return np.zeros(len(t))
    cum = np.concatenate([[0], np.cumsum(ends - starts)])      # cum[k] = total of the first k intervals
    k = np.searchsorted(ends, t, side="right")                 # intervals completely finished by t
    partial = np.where(k < len(starts), np.clip(t - starts[np.minimum(k, len(starts) - 1)], 0, None), 0)
    return cum[k] + partial


def make_edges(first, last, freq: str, tz: str = TZ) -> pd.DatetimeIndex:
    """Bin edges covering first..last (inclusive of the bin that holds `last`), tz-aware."""
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    first = first.tz_localize("UTC") if first.tzinfo is None else first.tz_convert("UTC")
    last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
    if freq in ("D", "1D"):
        lo = first.tz_convert(tz).tz_localize(None).normalize()
        hi = last.tz_convert(tz).tz_localize(None).normalize() + pd.Timedelta(days=1)
        return pd.date_range(lo, hi, freq="D").tz_localize(tz)    # local midnights: 23 or 25 h on clock-change days
    lo = first.floor(freq)
    hi = last.floor(freq) + pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(lo, hi, freq=freq).tz_convert(tz)


def minutes_by_bin(alerts: pd.DataFrame, freq: str = "D", as_of=None, start=None, end=None, tz: str = TZ) -> pd.DataFrame:
    """Minutes under alert and number of alert starts per bin.

    freq : "D" (local calendar day), "h", "15min", ...
    as_of: tz-aware time; open alerts count up to it and bins after it are marked incomplete.
    start, end: optional first and last time to cover (default: first alert and as_of / last end).
    Returns a DataFrame indexed by bin start (tz-aware) with columns:
      minutes      minutes under alert inside the bin (alerts clipped at the bin edges)
      n_starts     alerts that started in the bin
      bin_minutes  length of the bin itself (1440 for a normal day, 1380/1500 on clock-change days, 60 for an hour);
                   minutes / bin_minutes is the share of time under alert and minutes never exceeds bin_minutes
      complete     False for bins that end after `as_of` (still running); use only complete bins as actuals
    """
    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    starts, ends = _merged_intervals(alerts, as_of)
    first = pd.Timestamp(start) if start is not None else alerts["started_at"].min()
    if end is not None:
        last = pd.Timestamp(end)
    elif as_of is not None:
        last = as_of
    else:
        last = alerts["finished_at"].max()
    edges = make_edges(first, last, freq, tz)
    t = _ns(edges)
    F = _cumulative(starts, ends, t)
    minutes = np.diff(F) / _NS_PER_MIN
    bin_minutes = np.diff(t) / _NS_PER_MIN
    # starts per bin: count each alert start in the bin that contains it
    st = alerts["started_at"]
    if as_of is not None:
        st = st[st < as_of]
    pos = np.searchsorted(t, _ns(st), side="right") - 1
    pos = pos[(pos >= 0) & (pos < len(t) - 1)]
    n_starts = np.bincount(pos, minlength=len(t) - 1)
    complete = np.ones(len(minutes), dtype=bool) if as_of is None else (t[1:] <= as_of.value)
    out = pd.DataFrame({"minutes": minutes, "n_starts": n_starts, "bin_minutes": bin_minutes, "complete": complete},
                       index=edges[:-1])
    out.index.name = "bin_start"
    return out
