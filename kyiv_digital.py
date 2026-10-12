"""Fetch and parse the Kyiv Digital air-alert history page into a pandas DataFrame.

Page: https://kyiv.digital/storage/air-alert/stats.html
Rows are newest-first; each row is an event: alert start (🔴) or all-clear (🟢),
local Kyiv time as 'HH:MM DD.MM.YY', duration only on all-clear rows.
Every download is the FULL history, so the newest snapshot supersedes older ones.

Usage (run from anywhere; snapshots live next to this script):
    python3 kyiv_digital.py fetch                         # download + save snapshots/stats_<UTC time>.html
    python3 kyiv_digital.py parse [PATH ...] [--out F]    # PATH = files and/or folders; default = snapshots/
    python3 kyiv_digital.py check [PATH ...]              # self-consistency checks on a snapshot

If several snapshots are given (e.g. a shell glob) the newest one is used.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd

URL = "https://kyiv.digital/storage/air-alert/stats.html"
TZ = "Europe/Kyiv"
START_MARK, END_MARK = "🔴", "🟢"
HERE = Path(__file__).resolve().parent
SNAP_DIR = HERE / "snapshots"
STAMP_RE = re.compile(r"stats_(\d{8}T\d{6}Z)\.html$")


# --------------------------------------------------------------------------- snapshots
def validate_page(text: str) -> None:
    """Raise ValueError unless the text looks like the alert-history page."""
    if "<table" not in text or START_MARK not in text or not parse_summary(text):
        raise ValueError("response does not look like the Kyiv Digital alert page "
                         "(no table / markers / header line) - not saved")


def fetch(out_dir: str | Path | None = None) -> Path:
    """Download the page once and save it with the fetch time in the file name.

    The fetch time matters for leakage control: an alert's end is only 'known'
    from the first snapshot in which its all-clear row appears.
    The page is validated first and written atomically, so a bad or partial
    download can never become the 'newest snapshot'.
    """
    import requests

    resp = requests.get(URL, timeout=30,
                        headers={"User-Agent": "kyiv-alert-research/0.1 (student time-series project)"})
    resp.raise_for_status()
    resp.encoding = "utf-8"
    validate_page(resp.text)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(out_dir) if out_dir else SNAP_DIR
    path.mkdir(parents=True, exist_ok=True)
    final = path / f"stats_{stamp}.html"
    tmp = path / f"stats_{stamp}.html.part"   # does not match the stats_*.html pattern
    tmp.write_text(resp.text, encoding="utf-8")
    tmp.replace(final)
    return final


def snapshot_time(path: Path) -> datetime:
    """UTC time of a snapshot: from the file name if it has our stamp, else file modification time."""
    m = STAMP_RE.search(path.name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def resolve_snapshot(paths: list[str | Path] | None = None) -> Path:
    """Pick the snapshot to use from files and/or folders (default: the snapshots folder).

    With several candidates the newest wins, because each snapshot is the full history.
    """
    files: list[Path] = []
    for p in map(Path, paths or [SNAP_DIR]):
        if p.is_dir():
            files += sorted(p.glob("stats_*.html"))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"{p} does not exist")
    if not files:
        raise FileNotFoundError("no snapshot found - run 'fetch' first")
    newest = max(files, key=snapshot_time)
    if len(files) > 1:
        print(f"{len(files)} snapshots found; using the newest: {newest.name}", file=sys.stderr)
    return newest


# --------------------------------------------------------------------------- parsing
def parse_duration_minutes(text: str | None) -> float | None:
    """'1 годину 9 хвилин' -> 69, '31 хвилину' -> 31, '2 години' -> 120, '6 секунд' -> 0.1."""
    if not isinstance(text, str):
        return None
    t = text.lower()
    days = re.search(r"(\d+)\s*(?:доб|дн|день)", t)
    hours = re.search(r"(\d+)\s*годин", t)
    mins = re.search(r"(\d+)\s*хвилин", t)
    secs = re.search(r"(\d+)\s*секунд", t)
    if not (days or hours or mins or secs):
        return None
    return (int(days.group(1)) * 1440 if days else 0) + \
           (int(hours.group(1)) * 60 if hours else 0) + \
           (int(mins.group(1)) if mins else 0) + \
           (int(secs.group(1)) / 60 if secs else 0)


def _event_table(html: str) -> pd.DataFrame:
    """Return the table that holds the alert events (not just the first table on the page)."""
    for t in pd.read_html(StringIO(html), header=None):
        if t.shape[1] >= 2 and t.iloc[:, 1].astype(str).str.contains(f"{START_MARK}|{END_MARK}").any():
            return t
    raise ValueError("no table with alert events found - page layout may have changed")


def parse_events(html: str) -> pd.DataFrame:
    """One row per event, oldest first, with timestamps in UTC.

    Local times have no UTC offset and repeat once a year when clocks go back
    (03:00-03:59 happens twice). Because we walk the table in chronological order,
    an ambiguous time is resolved to the earliest reading that does not move
    backwards in time.
    """
    raw = _event_table(html).iloc[:, :3].copy()
    raw.columns = ["local", "label", "duration_text"][: raw.shape[1]]
    if "duration_text" not in raw:
        raw["duration_text"] = None

    raw = raw.iloc[::-1].reset_index(drop=True)  # oldest first
    kinds = raw["label"].astype(str).map(
        lambda s: "start" if START_MARK in s else ("end" if END_MARK in s else None))
    if kinds.isna().any():
        bad = raw.loc[kinds.isna(), "label"].unique()[:5]
        raise ValueError(f"unrecognised event labels: {list(bad)}")

    naive = pd.to_datetime(raw["local"].astype(str).str.strip(), format="%H:%M %d.%m.%y")
    utc, prev = [], None
    for ts in naive:
        # ambiguous=True is the first (summer-time) reading, False the second; for ordinary
        # times both give the same instant. Times skipped by the spring clock change cannot
        # occur in real data; shift_forward just keeps the parser from crashing on them.
        cands = sorted({ts.tz_localize(TZ, ambiguous=amb, nonexistent="shift_forward").tz_convert("UTC")
                        for amb in (True, False)})
        pick = next((c for c in cands if prev is None or c >= prev), cands[-1])
        utc.append(pick)
        prev = pick

    return pd.DataFrame({
        "ts_utc": utc,
        "kind": kinds.values,
        "page_duration_min": raw["duration_text"].map(parse_duration_minutes).values,
    })


def pair_alerts(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Turn the event stream into alert intervals and report anomalies.

    Anomalies: orphan_end (all-clear without a start), double_start (a start
    followed by another start, i.e. a missing all-clear), open_alert (last start
    has no end yet), duration_mismatch (computed vs. the page's own duration).
    """
    rows, anomalies = [], {"orphan_end": 0, "double_start": 0, "open_alert": 0, "duration_mismatch": 0}
    cur = None
    for e in events.itertuples(index=False):
        if e.kind == "start":
            if cur is not None:
                anomalies["double_start"] += 1
                rows.append({"started_at": cur, "finished_at": pd.NaT, "page_duration_min": None})
            cur = e.ts_utc
        else:
            if cur is None:
                anomalies["orphan_end"] += 1
                continue
            rows.append({"started_at": cur, "finished_at": e.ts_utc, "page_duration_min": e.page_duration_min})
            cur = None
    if cur is not None:
        anomalies["open_alert"] += 1
        rows.append({"started_at": cur, "finished_at": pd.NaT, "page_duration_min": None})

    out = pd.DataFrame(rows, columns=["started_at", "finished_at", "page_duration_min"])
    for c in ("started_at", "finished_at"):
        out[c] = pd.to_datetime(out[c], utc=True)  # keeps NaT tz-aware for open alerts
    out["page_duration_min"] = pd.to_numeric(out["page_duration_min"])
    out["duration_min"] = (out["finished_at"] - out["started_at"]).dt.total_seconds() / 60
    # The page truncates to whole minutes, so allow 1 minute of slack.
    both = out["page_duration_min"].notna() & out["duration_min"].notna()
    anomalies["duration_mismatch"] = int(((out.loc[both, "duration_min"] - out.loc[both, "page_duration_min"]).abs() > 1.01).sum())
    return out, anomalies


def parse_summary(html: str) -> dict:
    """Header line: 'З 24 лютого пролунало 2687 тривог. Небезпека тривала 3100 годин 17 хвилин.'"""
    m = re.search(r"пролунало\s+(\d+)\s+тривог.*?тривала\s+(\d+)\s+годин\w*\s+(\d+)\s+хвилин", html, re.S)
    return {"total_alerts": int(m.group(1)), "total_minutes": int(m.group(2)) * 60 + int(m.group(3))} if m else {}


def load(path: str | Path) -> tuple[pd.DataFrame, dict, dict]:
    html = Path(path).read_text(encoding="utf-8")
    alerts, anomalies = pair_alerts(parse_events(html))
    return alerts, anomalies, parse_summary(html)


# --------------------------------------------------------------------------- checks
def check_results(html: str) -> list[tuple[bool, bool, str]]:
    """Self-consistency checks on one snapshot -> list of (hard, ok, text).

    These cannot prove the page itself is right (that needs a cross-check against
    other sources), but they catch parser bugs and layout changes.
    """
    ev = parse_events(html)
    alerts, an = pair_alerts(ev)
    su = parse_summary(html)
    n_start = int((ev["kind"] == "start").sum())
    res: list[tuple[bool, bool, str]] = []
    if su:
        res.append((True, n_start == su["total_alerts"], f"start rows {n_start} == header alert count {su['total_alerts']}"))
    else:
        res.append((True, False, "header summary line not found (page layout changed?)"))
    res.append((True, bool(ev["ts_utc"].is_monotonic_increasing), "events are in chronological order (UTC)"))
    res.append((True, int((alerts["duration_min"] < 0).sum()) == 0, "no negative durations"))
    both = alerts.dropna(subset=["page_duration_min", "duration_min"])
    off = (both["duration_min"] - both["page_duration_min"]).abs() > 1.01
    share = 1 - off.mean() if len(both) else 0
    res.append((True, share >= 0.99, f"{share:.1%} of alerts match the page's own duration column within 1 min "
                                     f"({int(off.sum())} differ - timestamps are the more reliable of the two)"))
    after = alerts[alerts["started_at"] >= pd.Timestamp("2022-06-01", tz="UTC")]
    # An alert that is running right now legitimately has no all-clear yet - don't count it as broken.
    broken_late = int(after["finished_at"].isna().sum()) - an["open_alert"]
    res.append((False, broken_late == 0, f"alerts without an all-clear since 2022-06-01 (excluding one running now): {broken_late} "
                                         f"(whole history: {an['double_start']} missing all-clears, {an['orphan_end']} orphan all-clears)"))
    unparsed = int(ev[ev["kind"] == "end"]["page_duration_min"].isna().sum())
    res.append((False, unparsed == 0, f"all-clear rows whose duration text could not be read: {unparsed}"))
    res.append((False, an["open_alert"] == 0, "no alert running at snapshot time" if an["open_alert"] == 0 else
                "an alert was running at snapshot time (its end is unknown in this snapshot)"))
    return res


def check(html: str) -> bool:
    """Print the check results; True if every hard check passed."""
    res = check_results(html)
    for hard, ok, text in res:
        print(("PASS " if ok else ("FAIL " if hard else "WARN ")) + text)
    alerts, _ = pair_alerts(parse_events(html))
    print(f"alerts {len(alerts)} | {alerts['started_at'].min()} -> {alerts['started_at'].max()} (UTC)")
    return all(ok for hard, ok, _ in res if hard)


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the page and save a timestamped snapshot")
    f.add_argument("--out", default=None, help="folder for snapshots (default: snapshots/ next to this script)")
    p = sub.add_parser("parse", help="parse a snapshot into a CSV of alert intervals")
    p.add_argument("paths", nargs="*", help="snapshot files and/or folders (default: snapshots/); newest wins")
    p.add_argument("--out", default="kyiv_digital_alerts.csv")
    c = sub.add_parser("check", help="self-consistency checks on a snapshot")
    c.add_argument("paths", nargs="*", help="snapshot files and/or folders (default: snapshots/); newest wins")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "fetch":
            print(fetch(a.out))
            return 0
        snap = resolve_snapshot(a.paths)
        if a.cmd == "check":
            return 0 if check(snap.read_text(encoding="utf-8")) else 1
        alerts, anomalies, summary = load(snap)
        alerts.insert(0, "snapshot_utc", snapshot_time(snap).strftime("%Y-%m-%dT%H:%M:%SZ"))
        alerts.to_csv(a.out, index=False)
        print(f"snapshot {snap.name} -> {Path(a.out).resolve()}")
        print(f"{len(alerts)} alerts, {alerts.started_at.min()} -> {alerts.started_at.max()} (UTC)")
        print("anomalies:", anomalies)
        if summary:
            print(f"page summary: {summary['total_alerts']} alerts / {summary['total_minutes']} min | "
                  f"parsed: {len(alerts)} / {alerts['duration_min'].sum():.0f} min")
        return 0
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
