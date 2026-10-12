# Kyiv Digital air-alert history: parser, snapshots and dataset notes

Parses the public page <https://kyiv.digital/storage/air-alert/stats.html> (air-raid alerts in
**Kyiv City**, one row per alert start / all-clear) into a table of alert intervals for
time-series work. All figures below were measured on snapshot `stats_20261012T105432Z.html`
(fetched 2026-10-12 10:54 UTC, 2,690 alerts) unless stated otherwise.

## Quick start

The environment is managed with [uv](https://docs.astral.sh/uv/); `uv.lock` pins every package, so a fresh
clone on any machine installs exactly the same versions.

```bash
# 1. install uv once (macOS: brew install uv | Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh
#    | Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex")
# 2. then, in the project folder:
uv sync                                         # creates .venv with the locked dependencies (and a suitable Python)
uv run pytest                                   # 8 tests should pass
uv run python kyiv_digital.py fetch             # saves snapshots/stats_<UTC time>.html next to the script
uv run python kyiv_digital.py check             # self-consistency checks on the newest snapshot
uv run python kyiv_digital.py parse --out kyiv_digital_alerts.csv
```

- `uv run` uses the project's `.venv` automatically, so there is nothing to activate and no `python` / `python3` / `pip`
  naming to get wrong. In PyCharm, point the project interpreter at `.venv` (Settings, Project, Python Interpreter).
- Without uv, `pip install -r requirements.txt` works too. That file is **generated from `uv.lock`; do not edit it by
  hand** (`uv export --format requirements-txt --no-hashes --no-dev --no-emit-project -o requirements.txt`).
- Python 3.11 or newer is required (`.python-version` suggests 3.12). To change a dependency, edit `pyproject.toml`, run
  `uv lock`, then re-export `requirements.txt` and commit both. `uv lock --upgrade` moves everything to the newest versions.
- `parse` and `check` accept files and/or folders. With no argument they use `snapshots/`.
  If several snapshots are given (for example the glob `snapshots/stats_*.html`), **the newest is
  used**, because every download is the full history. The chosen file is printed.
- `fetch` validates the response and writes atomically; a maintenance page or partial download is
  never saved as a snapshot. **Keep every snapshot**: they are the only record of what the page said when.

## Output (`kyiv_digital_alerts.csv`)

| column | meaning |
|---|---|
| `snapshot_utc` | when the snapshot used was downloaded (from the file name); the earliest time at which everything in this file could have been known |
| `started_at` | alert start, **UTC**, truncated to the minute |
| `finished_at` | all-clear, UTC; **empty** if the page has no all-clear for this alert (see L10, L18) |
| `page_duration_min` | the page's own duration text converted to minutes (empty if absent) |
| `duration_min` | `finished_at - started_at` in minutes; empty if `finished_at` is empty. Prefer this to `page_duration_min` (L12) |

## Reproducibility: what was actually tested

| Check | Result |
|---|---|
| Fresh copy (only the committed files, no `.venv`) then `uv sync --locked`, `pytest`, `check` on a real snapshot | passes on Python 3.11, 3.12 and 3.13 (Linux x86-64) |
| Same code with pandas 2.2.3 instead of the locked 3.0.6 | all tests pass; `parse` output is byte-identical (2,691 lines) |
| Plain `pip install -r requirements.txt` in a fresh venv | works |
| Machine with no system time-zone database | works because `tzdata` is a direct dependency. Without it, pandas only installs `tzdata` on Windows and the run fails with `No time zone found with key Europe/Kyiv` |
| macOS and Windows | **not tested by the author.** The lock is universal (it resolves for all platforms and these packages publish wheels for them), so `uv sync --locked && uv run pytest` is the quick confirmation on a new machine |

## Where the data comes from

The page is served from `kyiv.digital`, the domain of the Kyiv Digital city app. The Kyiv City
Administration's open-data set "Statistics of air alerts in Kyiv" (data.gov.ua) is described by the city
as built from the same app data. We do not know how the app itself obtains the alerts.

## Limitations (read before modelling)

### Source and access
- **L1 Unofficial, undocumented endpoint.** No schema, versioning, uptime promise or change log. The
  parser fails loudly on an unknown event label, missing table or missing header line, but it cannot
  detect a silent change in what a row *means*.
- **L2 No stated licence or terms.** `robots.txt` allows all paths, but the page says nothing about reuse.
  Fetch rarely, cache snapshots, attribute the source. For bulk or redistributable use prefer the city's
  data.gov.ua dataset, which has explicit free-reuse terms (updated weekly; **not yet inspected by us**).
- **L3 Not independent ground truth.** It agrees with the other public sources (see Validation) but
  probably shares upstream announcements with them, so agreement shows consistency, not truth. Nobody
  here has measured when sirens actually sounded.

### Content and scope
- **L4 Kyiv City only.** No oblast or raion rows and no neighbouring regions. Earlier work on this problem
  found neighbouring-region alerts to be a strong predictor; they must come from another source.
- **L5 No alert level or threat type.** No yellow/red level (the two-level system was announced on
  2 Sep 2026), no weapon type, direction or explosion reports. Those need alerts.in.ua or Telegram data.
- **L6 Only two event types.** An alert is "start until all-clear". A mid-alert escalation (for example yellow to
  red) or a pause is invisible.
- **L7 Minute resolution, no stated time zone.** Times are truncated to the minute (median −32 s against the
  official CSV, as expected from truncation). We assume Europe/Kyiv; the offsets against the official CSV
  confirm it, because a wrong zone would show as whole hours. Do not compare at seconds level.
- **L8 Clock change on 25 Oct 2026.** Local times carry no UTC offset, so 03:00–03:59 occurs twice that
  night. The parser resolves it using the page's chronological order. This is tested only on synthetic
  data: no alert in the real history falls inside an ambiguous hour of the 2022–2025 changes.
  **After 25 Oct 2026, inspect alerts between 00:00 and 01:00 UTC that night.**

### Completeness and quality
- **L9 The first months are unreliable.** 47 alerts have no all-clear (17 in Feb 2022, 28 in Mar, 1 in Apr,
  1 on 28 Jun 2022) and 6 all-clears have no start (25 Mar to 31 May 2022), so at least 6 alerts are missing
  entirely. The first listed alert is 2022-02-25 05:19 UTC although the page says "since 24 February".
  **Suggested training start: 2022-07-01** (about 2,250 alerts after 29 Jun 2022).
- **L10 Alerts without an end.** Outside that early period, exactly one alert (28 Jun 2022) has no all-clear.
  `duration_min` is empty for these; do not fill with zero.
- **L11 Header totals do not reconcile.** The page says 186,296 minutes in total, but its duration column sums to
  184,991 and the timestamps to 181,553. We cannot tell how the page computes its total, so only the alert
  count (2,690, which matches exactly) is used as a check.
- **L12 The page's duration column is sometimes wrong.** 3 of 2,643 alerts disagree with the timestamps by
  more than a minute (for example 144 against 36, and 76 against 2). The official CSV agrees with the timestamps.
  Durations may be written in seconds ("6 секунд"); the parser handles this.
- **L13 "One alert" is the source's decision.** Merging and splitting rules are unknown. Restarts within 5 min
  of the previous all-clear are rare (0 to 3% per month in 2026). The longest alerts last 764, 673 and 632 minutes.

### Time, drift and leakage
- **L14 The page is a mutable full history.** Past rows could be edited without notice. We have one real
  snapshot, so we have never compared two. Keep all snapshots and diff them.
- **L15 Publication time is not recorded.** Rows have event times, not the time the page was updated, and the lag
  between a siren and its row is unmeasured. A snapshot only proves an all-clear was known *by its fetch time*.
  Backtests that treat `finished_at` as known at `finished_at` are therefore optimistic. Use only alerts with
  `started_at <= t`, and treat their end as unknown if it is after `t`.
- **L16 Possible regime change in recent months.** Alerts per month: 35 to 58 (Feb to Jun 2026), then
  59 (Jul), 121 (Aug), 190 (Sep) and 91 in the first 12 days of Oct. Median duration rose from 23–25 min
  (May to Jul) to 31, 40 and 33 min (Aug, Sep, Oct). Alerts of 5 min or less were 0–3% of the total in Feb to
  Jun, and 5%, 7% and 8% in Jul, Sep and Oct (2% in Aug). This may be real escalation, or a change in how alerts are issued or recorded (the second siren for
  the all-clear was dropped and the yellow/red system started in Sep 2026). We cannot separate the two:
  validate with time-ordered splits and consider a regime indicator.
- **L17 The newest weeks are only weakly verified.** The official CSV ends on 5 Sep 2026. After that date the
  volunteer CSV has 153 alerts against this page's 243, and only 24% to 39% of starts match. One day (11 Oct)
  agrees with alerts.in.ua: 8 alerts, 5 h 25 min here against 8 alerts, 5 h 24 min there.
- **L18 The last row may be a running alert.** Its `finished_at` is empty until the all-clear is published
  (`check` reports it as a WARN, which is normal).

### Pipeline
- **L19 Parser assumptions.** Table layout, the 🔴 and 🟢 markers, Ukrainian duration wording and the header
  sentence ("пролунало N тривог ..."). A new duration wording becomes an empty `page_duration_min` and is
  reported by `check`.
- **L20 Manual, single-shot collection.** One `fetch` is one snapshot; nothing runs on a schedule and the CSV
  is regenerated, not versioned (it records `snapshot_utc`). The cloud sandbox where this was developed
  cannot reach the site (HTTP 403 from its proxy); fetching works from your own machine.
- **L21 Small test base.** One real snapshot plus four synthetic fixtures (sample, clock change, odd rows, bad page).

## Validation performed

| Check | Result |
|---|---|
| Alert count against the page header | 2,690 = 2,690 |
| Events in time order, no negative durations | yes |
| Start times against `official_data_en.csv` (Kyiv City, 2022-06-01 to 2026-09-05) | 97.5% of this page's starts have an official start within 5 min; 99.8% the other way (2 min: 97.0% / 99.2%) |
| Time-zone conversion | median start offset −32 s, 5th to 95th percentile −60 to −4 s |
| Durations against the official file (2,006 matched alerts) | median difference 0.3 min; 99.5% within 3 min; correlation 0.998 |
| Yearly counts 2023 / 2024 / 2025 (this page / official) | 300 / 301, 501 / 502, 511 / 511 |
| Volunteer CSV, same window | 77% / 74% start match: the noisier source |
| 11 Oct 2026 against the alerts.in.ua report | 8 alerts, 325 min against 8 alerts, 324 min |

"Official" here means `datasets/official_data_en.csv` from `Vadimkin/ukrainian-air-raid-sirens-dataset`, which that
repo builds from the Telegram channel `@air_alert_ua`. It is not a government dataset (see L3).
The comparison used the nearest start within the tolerance, so a stray alert near another one can pair with the wrong partner.

## Still to do
1. Download and compare the data.gov.ua JSON (L2, L3).
2. Re-run `check` and inspect the night of 25 Oct 2026 (L8).
3. Take regular snapshots and diff them (L14, L15).
4. Build the hourly Kyiv series from 2022-07-01 and decide how to treat the Aug–Oct 2026 regime (L16).
