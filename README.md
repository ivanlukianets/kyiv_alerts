# Kyiv air-raid alerts: a 7-day forecast of minutes under alert

How many minutes will Kyiv City spend under air-raid alert on each of the next 7 days? This project answers that with a
point forecast and 80% and 95% intervals per day, using only information that was public before the forecast is made.
It parses the alert history published by Kyiv Digital, adds weather and a holiday calendar, explores the series, runs a
rolling-origin backtest of baselines and models, and makes the forecast. Everything is in one notebook, `analysis.ipynb`.

## Results at a glance

Snapshot of the alert page taken on 12 Oct 2026 at 11:02 UTC; complete days up to 11 Oct 2026.

**Forecast for 13 to 19 Oct 2026** (`results/forecast_20261013_arima_sqrt_all.csv`)

| Model | Minutes under alert per day | 80% interval | 95% interval |
|---|---|---|---|
| ARIMA(1,1,1) on the square root (chosen) | 437 | 57 to 1,440 | 30 to 1,440 |
| ARIMA + 7-day weighted average (alternative) | 471 | 47 to 1,440 | |

The forecast is flat over the week. The upper bounds reach a whole day (1,440 minutes): at the current level a full-alert
day cannot be ruled out.

**Backtest** (one forecast per day from 1 Jul 2024, horizons 1 to 7 days, scored on target days up to 27 Sep 2026,
the period the model was chosen on; MAE in minutes per day)

| Method | MAE, all test days | MAE since 27 Aug 2026 | 80% / 95% band coverage |
|---|---|---|---|
| ARIMA on the square root (chosen) | 101.5 | 339 | 79% / 93% |
| ETS on the square root | 101.5 | 337 | |
| ARIMA + 7-day weighted average | 103.6 | 319 | 79% / 93% |
| Ridge regression, best version | 104.9 | 348 | |
| 30-day mean | 109.5 | 309 | |
| Gradient-boosted trees, best version | 112.1 | 357 | |
| Yesterday's value | 136.5 | 335 | |

The last 14 days (28 Sep to 11 Oct) are kept for the confirmation below. Coverage is for the interval settings chosen in
section 6d of the notebook.

**Main findings**
- The level of alert minutes shifted twice: about 64 minutes per day until 10 Aug 2024, 114 until 26 Aug 2026 and
  485 since 27 Aug 2026. Old data helps only while the level stays put.
- Simple level forecasters (ETS, ARIMA, 14- and 30-day medians) beat models with more inputs. Weather and holidays add
  nothing measurable.
- The day-to-day spread grows with the level, so the intervals are built on the scale minutes ** 0.3 and widen when the
  level is high.
- On the last 14 days (the frozen choice, two weekly forecasts) the chosen model has MAE 270 and forecasts too low
  (bias -101); 13 of 14 days fall inside its 80% band.

Caveats: the model choice is a near tie and was made on the same test period it is reported on; the newest level has only
46 days of history. Section 8 of the notebook lists all caveats.

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/). `uv.lock` pins every package, so a fresh clone installs
exactly the same versions (tested on Linux and macOS; Windows should work but is untested). No system libraries are needed.

Install uv once (pick your system):

```bash
brew install uv
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then copy this block and run it in one go. It clones the repo, installs the locked environment, runs the tests and opens
the notebook in JupyterLab:

```bash
git clone https://github.com/ivanlukianets/kyiv_alerts.git
cd kyiv_alerts
uv sync
uv run pytest
uv run jupyter lab analysis.ipynb
```

- `uv run pytest` should report 49 passed.
- The committed notebook already shows all outputs. To reproduce them, choose Kernel, Restart Kernel and Run All Cells.
  The repo includes the snapshot, weather and holiday files it was run on, so the numbers come out the same. The first
  run computes the model backtest (about 5 minutes) and caches it in `results/`.
- `uv run jupyter lab` keeps running; stop it with Ctrl+C.
- `uv run` uses the project's `.venv` automatically, so there is nothing to activate. In PyCharm, point the project
  interpreter at `.venv`.
- Python 3.11 or newer (`.python-version` suggests 3.12; uv installs it if needed).
- Without uv: `pip install -r requirements.txt` in a virtual environment. That file is generated from `uv.lock`
  (`uv export --format requirements-txt --no-hashes --no-emit-project -o requirements.txt`); do not edit it by hand.
  To change a dependency, edit `pyproject.toml`, run `uv lock`, re-export `requirements.txt` and commit all three.

### Make a new forecast

Download a fresh alert page and weather, ideally shortly after midnight Kyiv time so that the day before the forecast is
complete, then rerun the notebook:

```bash
uv run python kyiv_digital.py fetch
uv run python kyiv_digital.py check
uv run python weather.py fetch
uv run jupyter lab analysis.ipynb
```

- `fetch` saves `snapshots/stats_<UTC time>.html`; the notebook always uses the newest snapshot. `check` runs
  self-consistency checks (a WARN for a running last alert is normal).
- Forecast origin: a snapshot taken before 06:00 Kyiv time forecasts from that day; a later snapshot forecasts from the
  next midnight.
- The model choice is frozen (`FROZEN_WINNER` in section 6b); set it to `None` to re-choose on the new data.
- The written readings in the notebook are dated to the 12 Oct 2026 snapshot; tables and plots update, the text does not.
- The holiday file covers dates up to 31 Dec 2026. For later forecasts run
  `uv run python holiday_calendar.py build --end 2027-12-31` first.
- Optional, the parsed alerts as a CSV: `uv run python kyiv_digital.py parse --out kyiv_digital_alerts.csv`.

## Repository layout

| Path | What it is |
|---|---|
| `analysis.ipynb` | the analysis: data, exploration, backtest, models, forecast (paired with `analysis.py`) |
| `analysis.py` | the same notebook as plain text (jupytext), for readable diffs |
| `kyiv_digital.py` | downloads, parses and checks the alert page |
| `series.py` | alert intervals to minutes under alert per day or hour, with leakage control |
| `explore.py` | exploration helpers: change points, autocorrelation, heavy-day clustering, spread against level |
| `backtest.py` | rolling-origin backtest, baselines, metrics (MAE, MASE, RMSE, bias, interval scores), Diebold-Mariano test |
| `models.py` | ETS, ARIMA, ridge, gradient-boosted trees, forecast combinations, level-scaled intervals, final forecast |
| `weather.py` | downloads daily Kyiv weather from Open-Meteo |
| `holiday_calendar.py` | builds the holiday and symbolic-date calendar from rules |
| `snapshots/` | downloaded alert pages, one file per download (keep them all) |
| `weather_kyiv_daily.csv` | weather input used by the notebook |
| `holiday_events.csv`, `holiday_features_daily.csv` | holiday inputs used by the notebook |
| `results/` | forecast tables (committed) and the cached model backtest (`*.pkl`, not committed) |
| `tests/` | 49 tests, including leakage tests for every forecasting step |

The notebook is paired with `analysis.py` by jupytext: edit either one, then run `uv run jupytext --sync analysis.ipynb`.
Commit both files; the notebook is committed with its outputs so the results can be read on GitHub.

## Datasets

All three inputs are joined by local calendar date (Europe/Kyiv). Nothing but the Kyiv Digital page is used as alert data.

### 1. Alert history: Kyiv Digital

| | |
|---|---|
| Source | <https://kyiv.digital/storage/air-alert/stats.html>, a public page of the Kyiv Digital city app |
| Content | every air-raid alert in **Kyiv City**: start time and all-clear time, one row per event, full history on every download |
| Coverage | 25 Feb 2022 to the download time; the committed snapshot (12 Oct 2026, 11:02 UTC) has 2,690 alerts |
| Usable from | 1 Jul 2022 (the first months have missing all-clears and starts, see L9) |
| Time | local Kyiv time on the page, converted to UTC by the parser; minute resolution |
| Licence | none stated (see L2); fetch rarely, keep snapshots, attribute the source |
| File | `snapshots/stats_20261012T110244Z.html` (committed) |

Parsed alerts (`kyiv_digital.py parse`, also what the notebook uses):

| Column | Meaning |
|---|---|
| `snapshot_utc` | when the snapshot was downloaded; the earliest time at which everything in it could have been known |
| `started_at` | alert start, UTC, truncated to the minute |
| `finished_at` | all-clear, UTC; empty if the page has no all-clear for this alert (L10, L18) |
| `page_duration_min` | the page's own duration text in minutes (empty if absent) |
| `duration_min` | `finished_at - started_at` in minutes; prefer this to `page_duration_min` (L12) |

Daily series built in the notebook (`series.minutes_by_bin`, one row per local day, 1 Jul 2022 to the snapshot day):

| Column | Meaning |
|---|---|
| `minutes` | minutes under alert inside the day; an alert across midnight counts in both days; overlapping alerts are merged |
| `n_starts` | alerts that started that day |
| `bin_minutes` | length of the day: 1,440, or 1,380 / 1,500 on clock-change days |
| `complete` | `False` for the day still running at the snapshot time; only complete days are used as actual values |

Only the newest alert may still be running (counted up to the snapshot time); an older alert without an all-clear is left
out of the minutes rather than given a made-up end.

### 2. Weather: Open-Meteo

| | |
|---|---|
| Source | [Open-Meteo](https://open-meteo.com/), Historical Forecast API for past days and Forecast API for the coming days |
| Location | Kyiv, 50.45 N, 30.52 E, daily values in Europe/Kyiv time |
| Coverage | 1 Jul 2022 to 19 Oct 2026 in the committed file (1,572 days; 1,564 archived, 8 live forecast) |
| Licence | free for non-commercial use, CC BY 4.0: "Weather data by Open-Meteo.com" |
| File | `weather_kyiv_daily.csv` (committed; raw API responses go to `weather_raw/`, not committed) |

Columns: `weather_code`; temperature max, min, mean and apparent mean (°C); `sunrise`, `sunset`; precipitation, rain and
snowfall sums and precipitation hours; maximum wind speed and gusts (km/h) and dominant wind direction; mean cloud cover,
relative humidity, sea-level pressure and visibility; `daylight_min` and `sunshine_min` (minutes); `source`
(`historical_forecast` or `forecast`) and `fetched_utc`. The models use temperature, precipitation, wind, cloud cover and
daylight on the target day.

Caveat: the archive stitches together the first hours of successive model runs, so the value for day D is about a
forecast issued on day D. For forecast horizons 2 to 7 this is better than what was known at the origin, so backtest gains
from weather are optimistic (weather still did not help).

### 3. Holiday and symbolic-date calendar

| | |
|---|---|
| Source | hand-written rules in `holiday_calendar.py` (no network); Orthodox Easter computed with the Meeus algorithm |
| Coverage | 1 Jul 2022 to 31 Dec 2026, 111 events |
| Files | `holiday_events.csv` (date, country, name, kind, movable, rule) and `holiday_features_daily.csv` (one row per day) |

Groups: `ua_statehood`, `ua_military`, `ru_state_military`, `religious`, `new_year`, `war_anniversary` (plus `civil`,
listed but not used as features). For each group the daily file has `_is_day`, `_eve`, `_within2`, `_days_to_next` and
`_days_since_last` (capped at 14). It is not an official source and does not record days off (martial law changed them).
Each date occurs only about four times in the data, so the features are pooled by group.

## Method in short

- **Target:** minutes under alert per local day. **Horizon:** 7 days. **Forecast origin:** a local midnight; the
  forecaster sees only complete days before it.
- **Leakage control:** features, model fits, combination weights and interval errors use only data available at the
  origin. Tests change the future and check that past forecasts do not move.
- **Exploration** (section 4) uses only data before 1 Jul 2024; four checks on the full history (level shifts, heavy-day
  clustering, count against duration, spread against level) decide the kind of model and the interval scale.
- **Backtest:** one forecast per day from 1 Jul 2024, horizons 1 to 7; MAE as headline (MASE, RMSE, bias also reported),
  interval coverage and interval score, results per level slice, Diebold-Mariano tests with Newey-West errors.
- **Methods:** baselines (yesterday, same weekday, trailing means and medians over 7 days to the whole history,
  exponentially weighted means); ETS and ARIMA on the square root; ridge and gradient-boosted trees (Poisson loss) with and
  without weather and holidays; combinations of a smoother and a fast average. Training windows from 30 days to the whole
  history are compared.
- **Choice and confirmation:** lowest MAE on test targets before the last 14 days, frozen, then scored on the last 14 days.
- **Intervals:** each method's own past errors on the scale minutes ** 0.3 (from the spread-against-level slope 0.70),
  60-day look-back, capped at 1,440 minutes.

## Reproducibility: what was tested

| Check | Result |
|---|---|
| Clean copy of the repo files, `uv sync --locked`, `pytest`, full notebook run with no cache (Linux x86-64, Python 3.12) | 49 tests pass; the notebook runs end to end and reproduces the forecast |
| Full notebook on macOS (Apple Silicon) after `uv sync` | runs; the forecast file it wrote matches the Linux run |
| Parser with pandas 2.2.3 instead of the locked 3.0.6 | all parser tests pass; output byte-identical |
| Machine without a system time-zone database | works, because `tzdata` is a direct dependency |
| Windows | not tested; the lock is universal, so `uv sync --locked` then `uv run pytest` is the quick check |
| Network downloads (`kyiv_digital.py fetch`, `weather.py fetch`) | work from a personal machine; the test suite uses saved pages and a fake API |

## Data limitations (read before modelling)

### Source and access
- **L1 Unofficial, undocumented endpoint.** No schema, versioning, uptime promise or change log. The
  parser fails loudly on an unknown event label, missing table or missing header line, but it cannot
  detect a silent change in what a row *means*.
- **L2 No stated licence or terms.** `robots.txt` allows all paths, but the page says nothing about reuse.
  Fetch rarely, cache snapshots, attribute the source. For bulk or redistributable use prefer the city's
  data.gov.ua dataset ("Statistics of air alerts in Kyiv"), which has explicit free-reuse terms (updated weekly;
  **not yet inspected by us**). The city describes it as built from the same app data.
- **L3 Not independent ground truth.** It agrees with the other public sources (see Validation) but
  probably shares upstream announcements with them, so agreement shows consistency, not truth. Nobody
  here has measured when sirens actually sounded.

### Content and scope
- **L4 Kyiv City only.** No oblast or raion rows and no neighbouring regions. Earlier work on this problem
  found neighbouring-region alerts to be a strong predictor; they must come from another source.
- **L5 No alert level or threat type.** No yellow/red level (the two-level system was announced on
  2 Sep 2026), no weapon type, direction or explosion reports.
- **L6 Only two event types.** An alert is "start until all-clear". A mid-alert escalation or a pause is invisible.
- **L7 Minute resolution, no stated time zone.** Times are truncated to the minute (median −32 s against the
  official CSV, as expected from truncation). We assume Europe/Kyiv; the offsets against the official CSV
  confirm it, because a wrong zone would show as whole hours.
- **L8 Clock change on 25 Oct 2026.** Local times carry no UTC offset, so 03:00–03:59 occurs twice that
  night. The parser resolves it using the page's chronological order, tested only on synthetic data.
  **After 25 Oct 2026, inspect alerts between 00:00 and 01:00 UTC that night.**

### Completeness and quality
- **L9 The first months are unreliable.** 47 alerts have no all-clear (17 in Feb 2022, 28 in Mar, 1 in Apr,
  1 on 28 Jun 2022) and 6 all-clears have no start (25 Mar to 31 May 2022), so at least 6 alerts are missing
  entirely. The first listed alert is 2022-02-25 05:19 UTC although the page says "since 24 February".
  **The analysis starts on 2022-07-01.**
- **L10 Alerts without an end.** Outside that early period, exactly one alert (28 Jun 2022) has no all-clear.
  `duration_min` is empty for these; do not fill with zero.
- **L11 Header totals do not reconcile.** The page says 186,296 minutes in total, but its duration column sums to
  184,991 and the timestamps to 181,553. Only the alert count (2,690, which matches exactly) is used as a check.
- **L12 The page's duration column is sometimes wrong.** 3 of 2,643 alerts disagree with the timestamps by
  more than a minute. The official CSV agrees with the timestamps. Durations may be written in seconds; the parser handles this.
- **L13 "One alert" is the source's decision.** Merging and splitting rules are unknown. Restarts within 5 min
  of the previous all-clear are rare (0 to 3% per month in 2026). The longest alerts last 764, 673 and 632 minutes.

### Time, drift and leakage
- **L14 The page is a mutable full history.** Past rows could be edited without notice. The only two snapshots compared
  so far (12 Oct 2026, 10:54 and 11:02 UTC) are identical but 8 minutes apart. Keep all snapshots and diff them over days.
- **L15 Publication time is not recorded.** The lag between a siren and its row is unmeasured. A snapshot only proves an
  all-clear was known *by its fetch time*. Backtests that treat `finished_at` as known at `finished_at` are therefore
  optimistic; this project forecasts from midnight with complete days only.
- **L16 Possible regime change in recent months.** Alerts per month: 35 to 58 (Feb to Jun 2026), then 59 (Jul), 121 (Aug),
  190 (Sep) and 91 in the first 12 days of Oct. Median duration rose from 23–25 min (May to Jul) to 31, 40 and 33 min
  (Aug, Sep, Oct). This may be real escalation or a change in how alerts are issued or recorded (the second siren for the
  all-clear was dropped and the yellow/red system started in Sep 2026). We cannot separate the two.
- **L17 The newest weeks are only weakly verified.** The official CSV ends on 5 Sep 2026. After that date the volunteer
  CSV has 153 alerts against this page's 243, and only 24% to 39% of starts match. One day (11 Oct) agrees with
  alerts.in.ua: 8 alerts, 5 h 25 min here against 8 alerts, 5 h 24 min there.
- **L18 The last row may be a running alert.** Its `finished_at` is empty until the all-clear is published.

### Pipeline
- **L19 Parser assumptions.** Table layout, the 🔴 and 🟢 markers, Ukrainian duration wording and the header sentence.
  A new duration wording becomes an empty `page_duration_min` and is reported by `check`.
- **L20 Manual, single-shot collection.** One `fetch` is one snapshot; nothing runs on a schedule.
- **L21 Small test base for the parser.** One real snapshot plus four synthetic fixtures.

## Validation of the alert data

| Check | Result |
|---|---|
| Alert count against the page header | 2,690 = 2,690 |
| Events in time order, no negative durations | yes |
| Start times against `official_data_en.csv` (Kyiv City, 2022-06-01 to 2026-09-05) | 97.5% of this page's starts have an official start within 5 min; 99.8% the other way |
| Time-zone conversion | median start offset −32 s, 5th to 95th percentile −60 to −4 s |
| Durations against the official file (2,006 matched alerts) | median difference 0.3 min; 99.5% within 3 min; correlation 0.998 |
| Yearly counts 2023 / 2024 / 2025 (this page / official) | 300 / 301, 501 / 502, 511 / 511 |
| Volunteer CSV, same window | 77% / 74% start match: the noisier source |
| 11 Oct 2026 against the alerts.in.ua report | 8 alerts, 325 min against 8 alerts, 324 min |

"Official" means `datasets/official_data_en.csv` from
[Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset), built from the
Telegram channel `@air_alert_ua`. It is not a government dataset (see L3). These sources were used for validation only,
never as model inputs.

## Further work

1. Regime-switching and pretrained time-series models; a multiple-comparison correction or Model Confidence Set.
2. Neighbouring-region alerts and threat types from another source (L4, L5).
3. Regular snapshots to measure revisions and publication lag (L14, L15), and a weekly forecast scored after its week.
4. Compare with the data.gov.ua dataset (L2, L3); inspect the night of 25 Oct 2026 (L8).
5. The onset question (chance of an alert in a given hour) and the duration question, on the same series and backtest.

## Attribution

Alert data: Kyiv Digital (<https://kyiv.digital>). Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).
